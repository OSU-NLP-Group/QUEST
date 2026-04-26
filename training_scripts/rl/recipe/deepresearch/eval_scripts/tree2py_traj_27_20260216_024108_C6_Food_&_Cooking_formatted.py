import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "holiday_restaurant_hours_2025"
TASK_DESCRIPTION = (
    "Identify four nationally recognized restaurant chains that were confirmed to be open on both "
    "Thanksgiving Day 2025 (November 27, 2025) and Christmas Day 2025 (December 25, 2025). For each restaurant chain, "
    "you must provide: 1. The name of the restaurant chain, 2. The specific operating hours for Thanksgiving Day 2025, "
    "including both the opening time and closing time in standard time format (e.g., '8:00 AM - 7:00 PM' or '11 a.m. - 9 p.m.'). "
    "The hours must be specific times, not 'varies by location' or 'call ahead.', 3. A URL from a reliable source "
    "(such as the restaurant's official website, TODAY.com, USA Today, People, Axios, or similar major news outlets) that verifies "
    "the Thanksgiving Day 2025 operating hours you provided, 4. Confirmation that the restaurant chain was also open on Christmas Day 2025 "
    "(you do not need to provide specific hours for Christmas, just confirmation that it was open). All four restaurant chains must be national chains "
    "with locations across multiple U.S. states. The information must pertain to the 2025 holiday season."
)

# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
TRUSTED_DOMAINS = {
    # Examples of major/credible outlets per rubric and similar
    "today.com",
    "usatoday.com",
    "people.com",
    "axios.com",
    "nbcnews.com",
    "abcnews.go.com",
    "cbsnews.com",
    "cnn.com",
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
    "forbes.com",
    "wsj.com",
    "nytimes.com",
    "washingtonpost.com",
    "yahoo.com",
    "msn.com",
    "time.com",
    "businessinsider.com",
    "theverge.com",
    "foodnetwork.com",
    "delish.com",
    "eater.com",
    "foxnews.com",
}


def _parse_domain(url: Optional[str]) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _slugify_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if not parsed.netloc:
            return False
        return True
    except Exception:
        return False


def _is_reliable_source(url: Optional[str], chain_name: Optional[str]) -> bool:
    """
    Heuristic reliability:
    - URL is valid AND
      - domain matches a trusted outlet; OR
      - domain appears to be the chain's official site (chain slug in domain).
    """
    if not _is_valid_url(url):
        return False
    host = _parse_domain(url)
    if not host:
        return False

    # Trusted outlets
    for d in TRUSTED_DOMAINS:
        if host == d or host.endswith("." + d):
            return True

    # Official site heuristic (chain slug included in domain)
    slug = _slugify_name(chain_name)
    if slug and slug in host:
        return True

    return False


def _normalize_ampm_text(s: str) -> str:
    # Normalize different AM/PM stylings: remove dots, compress spaces
    t = s.lower()
    t = t.replace(".", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_standard_time_format(time_str: Optional[str]) -> bool:
    """
    Accept formats like:
    - 8 AM, 8:00 AM, 11 a.m., 11:30 p.m., 12 pm, etc.
    """
    if not time_str or not isinstance(time_str, str):
        return False
    t = _normalize_ampm_text(time_str)
    # Match 1-12, optional :mm, with am/pm (no dots after normalization)
    pattern = r"\b(1[0-2]|0?[1-9])(:[0-5][0-9])?\s?(am|pm)\b"
    return re.search(pattern, t) is not None


FORBIDDEN_NON_SPECIFIC_PHRASES = [
    "varies by location",
    "hours may vary",
    "call ahead",
    "check your local",
    "check with your local",
    "check local",
    "participating locations",
    "select locations",
]


def _is_specific_hours(hours_text: Optional[str]) -> bool:
    """
    Hours must be specific times; if text includes non-specific disclaimers, fail.
    If no raw text is provided but opening & closing times exist, we still accept specificity here.
    """
    if not hours_text:
        return True
    h = hours_text.lower()
    return not any(phrase in h for phrase in FORBIDDEN_NON_SPECIFIC_PHRASES)


def _unique_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    res: List[str] = []
    for u in urls:
        if not _is_valid_url(u):
            continue
        if u not in seen:
            seen.add(u)
            res.append(u)
    return res


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RestaurantItem(BaseModel):
    chain_name: Optional[str] = None
    thanksgiving_hours_text: Optional[str] = None
    opening_time: Optional[str] = None
    closing_time: Optional[str] = None
    thanksgiving_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)
    christmas_open_confirmation: Optional[str] = None  # e.g., "open", "yes", or text snippet
    christmas_urls: List[str] = Field(default_factory=list)


class RestaurantsExtraction(BaseModel):
    restaurants: List[RestaurantItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_restaurants() -> str:
    return """
Extract up to four restaurant chains and their 2025 holiday information as explicitly stated in the answer.

For each restaurant chain, extract the following fields:
- chain_name: The restaurant chain name exactly as mentioned in the answer.
- thanksgiving_hours_text: The raw phrase in the answer that describes the Thanksgiving Day 2025 hours (e.g., "8:00 AM - 7:00 PM", "11 a.m. - 9 p.m."). If the answer specifies only generic disclaimers like "varies by location" or "call ahead", include that phrase here.
- opening_time: The Thanksgiving 2025 opening time as a standalone value (e.g., "8:00 AM", "11 a.m."). If not explicitly given, set null.
- closing_time: The Thanksgiving 2025 closing time as a standalone value (e.g., "7:00 PM", "9 p.m."). If not explicitly given, set null.
- thanksgiving_url: A single URL from the answer that is claimed to verify the Thanksgiving Day 2025 hours for that chain (from the restaurant's official website or a major outlet like TODAY.com, USA Today, People, Axios, etc.). If no such URL is given, set null.
- additional_urls: Any other URLs in the answer associated with this restaurant (e.g., brand websites, location pages, or other articles). If none, return an empty list.
- christmas_open_confirmation: A short phrase from the answer indicating the chain is open on Christmas Day 2025 (e.g., "open on Christmas", "open on December 25"). If not stated, set null.
- christmas_urls: Any URLs in the answer that support the Christmas Day 2025 open status for the chain. If none, return an empty list.

Rules:
1) Do not invent times or URLs. Only extract what is explicitly present in the answer.
2) If the answer uses "11 a.m." / "9 p.m." formatting, keep that style in opening_time/closing_time.
3) If the answer only states generic disclaimers like "varies by location" and does not give explicit times, set opening_time and closing_time to null.
4) Include URLs exactly as written in the answer (plain URLs or markdown links).

Return a JSON object with a top-level key 'restaurants' that is an array of up to four RestaurantItem objects as defined above.
"""


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_restaurant(
    evaluator: Evaluator,
    parent_node,
    item: RestaurantItem,
    idx: int,
) -> None:
    """
    Build verification subtree for a single restaurant (parallel aggregation, non-critical at restaurant level).
    """
    rest_node = evaluator.add_parallel(
        id=f"restaurant_{idx+1}",
        desc=f"Restaurant #{idx+1} verification",
        parent=parent_node,
        critical=False,  # Allow partial credit across restaurants
    )

    # ---------------- Chain Identity ----------------
    chain_identity_node = evaluator.add_parallel(
        id=f"restaurant_{idx+1}_chain_identity",
        desc="The restaurant chain name is provided and it is identified as a nationally recognized chain with locations across multiple U.S. states",
        parent=rest_node,
        critical=True,
    )

    # Name provided (existence check)
    name_provided = evaluator.add_custom_node(
        result=bool(item.chain_name and item.chain_name.strip()),
        id=f"restaurant_{idx+1}_name_provided",
        desc="Chain name is provided",
        parent=chain_identity_node,
        critical=True,
    )

    # National presence verification (URL-grounded if possible)
    national_chain_leaf = evaluator.add_leaf(
        id=f"restaurant_{idx+1}_national_presence",
        desc="Chain is a nationally recognized chain with locations across multiple U.S. states",
        parent=chain_identity_node,
        critical=True,
    )

    national_sources = _unique_urls(
        ([item.thanksgiving_url] if item.thanksgiving_url else [])
        + item.additional_urls
        + item.christmas_urls
    )

    await evaluator.verify(
        claim=f"The restaurant chain '{item.chain_name or ''}' is a nationally recognized chain with locations across multiple U.S. states in the United States.",
        node=national_chain_leaf,
        sources=national_sources if national_sources else None,
        additional_instruction=(
            "Accept if the source(s) clearly indicate the chain is nationwide or has locations across multiple U.S. states. "
            "Official brand 'Locations' page indicating multiple U.S. states counts as sufficient evidence. "
            "Articles that describe the chain as 'national' or 'nationwide' also count."
        ),
    )

    # ---------------- Thanksgiving Day 2025 ----------------
    tg_node = evaluator.add_parallel(
        id=f"restaurant_{idx+1}_thanksgiving_2025",
        desc=f"Thanksgiving Day 2025 operating information for Restaurant #{idx+1}",
        parent=rest_node,
        critical=True,
    )

    # Operating hours (opening, closing, specificity)
    op_hours_node = evaluator.add_parallel(
        id=f"restaurant_{idx+1}_operating_hours",
        desc="Specific operating hours for Thanksgiving Day 2025 are provided in standard 12-hour time format with AM/PM designation",
        parent=tg_node,
        critical=True,
    )

    # Opening time format
    opening_time_ok = evaluator.add_custom_node(
        result=_is_standard_time_format(item.opening_time),
        id=f"restaurant_{idx+1}_opening_time",
        desc="Opening time is provided in standard time format (e.g., '8:00 AM', '11 a.m.')",
        parent=op_hours_node,
        critical=True,
    )

    # Closing time format
    closing_time_ok = evaluator.add_custom_node(
        result=_is_standard_time_format(item.closing_time),
        id=f"restaurant_{idx+1}_closing_time",
        desc="Closing time is provided in standard time format (e.g., '9:00 PM', '7 p.m.')",
        parent=op_hours_node,
        critical=True,
    )

    # Specificity (not 'varies by location', etc.)
    specificity_ok = evaluator.add_custom_node(
        result=_is_specific_hours(item.thanksgiving_hours_text),
        id=f"restaurant_{idx+1}_hours_specificity",
        desc="The hours are stated as specific times, not as 'varies by location' or 'call ahead'",
        parent=op_hours_node,
        critical=True,
    )

    # URL reference + verification of hours
    url_ref_node = evaluator.add_parallel(
        id=f"restaurant_{idx+1}_url_reference_thanksgiving",
        desc="A valid URL from a reliable source verifies the Thanksgiving Day 2025 hours",
        parent=tg_node,
        critical=True,
    )

    # Thanksgiving URL provided
    thanksgiving_url_provided = evaluator.add_custom_node(
        result=_is_valid_url(item.thanksgiving_url),
        id=f"restaurant_{idx+1}_thanksgiving_url_provided",
        desc="A valid URL verifying Thanksgiving Day 2025 hours is provided",
        parent=url_ref_node,
        critical=True,
    )

    # Thanksgiving URL reliability check
    thanksgiving_url_reliable = evaluator.add_custom_node(
        result=_is_reliable_source(item.thanksgiving_url, item.chain_name),
        id=f"restaurant_{idx+1}_thanksgiving_url_reliable",
        desc="The Thanksgiving hours URL is from a reliable source (official site or major outlet)",
        parent=url_ref_node,
        critical=True,
    )

    # Verify that the given Thanksgiving hours are supported by the URL
    hours_supported_leaf = evaluator.add_leaf(
        id=f"restaurant_{idx+1}_hours_supported",
        desc="The Thanksgiving 2025 hours are supported by the cited URL",
        parent=url_ref_node,
        critical=True,
    )

    tg_open = item.opening_time or ""
    tg_close = item.closing_time or ""
    tg_chain = item.chain_name or ""
    hours_claim = (
        f"On Thanksgiving Day 2025 (November 27, 2025), {tg_chain} is open from {tg_open} to {tg_close}."
    )

    await evaluator.verify(
        claim=hours_claim,
        node=hours_supported_leaf,
        sources=item.thanksgiving_url if item.thanksgiving_url else None,
        additional_instruction=(
            "Verify the page is for Thanksgiving 2025 (or clearly the 2025 holiday season). "
            "The page must explicitly support these specific hours for Thanksgiving Day 2025. "
            "Allow minor formatting variations (e.g., '11 a.m.' vs '11 AM', hyphen vs 'to'). "
            "Do not accept pages that only say 'hours vary by location' without explicit hours."
        ),
    )

    # ---------------- Christmas Day 2025 status ----------------
    christmas_leaf = evaluator.add_leaf(
        id=f"restaurant_{idx+1}_christmas_open_2025",
        desc="Confirmation that the restaurant chain was open on Christmas Day 2025",
        parent=rest_node,
        critical=True,
    )

    christmas_sources = _unique_urls(
        (item.christmas_urls if item.christmas_urls else [])
        + ([item.thanksgiving_url] if item.thanksgiving_url else [])
    )

    await evaluator.verify(
        claim=f"The restaurant chain '{tg_chain}' was open on Christmas Day 2025 (December 25, 2025).",
        node=christmas_leaf,
        sources=christmas_sources if christmas_sources else None,
        additional_instruction=(
            "Confirm the chain was open on Christmas Day 2025. "
            "A 2025 holiday hours page or a credible article explicitly mentioning Christmas Day 2025 counts. "
            "Exact hours for Christmas are not required—only confirmation of being open."
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
    Evaluate an answer for the 2025 holiday restaurant hours task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Restaurants evaluated independently
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_restaurants(),
        template_class=RestaurantsExtraction,
        extraction_name="restaurants_extraction",
    )

    # Normalize to exactly 4 items (pad with empty if fewer)
    items = list(extracted.restaurants) if extracted and extracted.restaurants else []
    items = items[:4]
    while len(items) < 4:
        items.append(RestaurantItem())

    # Build verification for each restaurant
    verify_tasks = []
    for i, item in enumerate(items):
        verify_tasks.append(verify_single_restaurant(evaluator, root, item, i))

    # Run verifications (sequentially is fine; could use gather)
    for t in verify_tasks:
        await t

    return evaluator.get_summary()