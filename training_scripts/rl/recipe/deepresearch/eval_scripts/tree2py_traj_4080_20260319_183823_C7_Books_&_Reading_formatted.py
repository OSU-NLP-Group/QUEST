import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "celebrity_memoirs_2025"
TASK_DESCRIPTION = """
I am creating a reading list for a book club interested in celebrity memoirs from 2025. Identify exactly 4 celebrity memoirs that were published in 2025 and meet all of the following criteria:

1. Each memoir must be written by a celebrity or public figure (such as an entertainer, athlete, or media personality)
2. Each memoir must be published by one of the 'Big Five' publishers (Penguin Random House, HarperCollins, Simon & Schuster, Hachette Book Group, or Macmillan) or their imprints
3. Each memoir must be available in both hardcover and audiobook formats
4. The audiobook version of each memoir must be narrated by the author themselves
5. The 4 memoirs must collectively represent at least 3 different Big Five publishers
6. At least one memoir must have been published in Q1 2025 (January-March), and at least one must have been published in Q2-Q4 2025 (April-December)

For each memoir, provide:
- The book title
- The author's name
- The publisher (including specific imprint if applicable)
- The publication date
- A direct link to the publisher's official page for the book or a reliable book retailer page
- Confirmation that both hardcover and audiobook formats are available
- Confirmation that the audiobook is narrated by the author
"""


# -----------------------------------------------------------------------------
# Utilities: month parsing & Big Five mapping
# -----------------------------------------------------------------------------
MONTHS_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def parse_month_year(date_str: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse a date string to (month, year). Very forgiving parser for typical book metadata.
    Returns (None, None) if cannot parse.
    """
    if not date_str:
        return None, None

    s = date_str.strip()
    s_low = s.lower()

    # Year detection
    year = None
    m_year = re.search(r"\b(20\d{2})\b", s_low)
    if m_year:
        try:
            year = int(m_year.group(1))
        except Exception:
            year = None

    # Month by name
    for name, m in MONTHS_MAP.items():
        if re.search(rf"\b{name}\b", s_low):
            return m, year

    # ISO-like YYYY-MM-DD
    m_iso = re.search(r"\b(20\d{2})[-/\.](\d{1,2})[-/\.](\d{1,2})\b", s_low)
    if m_iso:
        try:
            year = int(m_iso.group(1))
            month = int(m_iso.group(2))
            if 1 <= month <= 12:
                return month, year
        except Exception:
            pass

    # MM/DD/YYYY or M/D/YYYY
    m_us = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", s_low)
    if m_us:
        try:
            month = int(m_us.group(1))
            year = int(m_us.group(3))
            if 1 <= month <= 12:
                return month, year
        except Exception:
            pass

    # MM/YYYY
    m_mmyyyy = re.search(r"\b(\d{1,2})[/-](20\d{2})\b", s_low)
    if m_mmyyyy:
        try:
            month = int(m_mmyyyy.group(1))
            year = int(m_mmyyyy.group(2))
            if 1 <= month <= 12:
                return month, year
        except Exception:
            pass

    return None, year


PRH_PATTERNS = {
    "penguin random house", "penguinrandomhouse", "prh",
    "random house", "alfred a. knopf", "knopf", "crown", "doubleday",
    "viking", "putnam", "g. p. putnam", "gp putnam",
    "riverhead", "ballantine", "dutton", "harmony", "ten speed press",
    "clarkson potter", "broadway books", "portfolio", "avery", "hogarth",
    "penguin press"
}
HARPER_PATTERNS = {
    "harpercollins", "harper collins", "harper", "william morrow",
    "dey street", "harperone", "ecco", "amistad", "harlequin", "hanover square press"
}
SANDS_PATTERNS = {
    "simon & schuster", "simon and schuster", "s&s", "atria", "gallery books",
    "scribner", "pocket books", "adams media", "threshold editions", "saga press"
}
HACHETTE_PATTERNS = {
    "hachette book group", "hachette", "little, brown", "little brown",
    "grand central publishing", "mulholland books", "orbit", "twelve",
    "basic books", "running press", "seal press", "perseus books"
}
MACMILLAN_PATTERNS = {
    "macmillan", "us.macmillan", "st. martin", "st martin", "henry holt",
    "farrar, straus and giroux", "farrar straus", "fsg", "flatiron",
    "tor", "forge", "minotaur", "picador", "celadon"
}

DOMAIN_TO_GROUP = {
    "penguinrandomhouse.com": "Penguin Random House",
    "prh.com": "Penguin Random House",
    "harpercollins.com": "HarperCollins",
    "simonandschuster.com": "Simon & Schuster",
    "hachettebookgroup.com": "Hachette Book Group",
    "littlebrown.com": "Hachette Book Group",
    "grandcentralpublishing.com": "Hachette Book Group",
    "orbitbooks.net": "Hachette Book Group",
    "us.macmillan.com": "Macmillan",
    "macmillan.com": "Macmillan",
    "stmartins.com": "Macmillan",
    "fsgoriginals.com": "Macmillan",
    "tor.com": "Macmillan",
}

GROUP_PATTERNS = [
    ("Penguin Random House", PRH_PATTERNS),
    ("HarperCollins", HARPER_PATTERNS),
    ("Simon & Schuster", SANDS_PATTERNS),
    ("Hachette Book Group", HACHETTE_PATTERNS),
    ("Macmillan", MACMILLAN_PATTERNS),
]


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def infer_big_five_group_from_publisher(publisher_name: Optional[str]) -> Optional[str]:
    if not publisher_name:
        return None
    p = normalize_text(publisher_name)
    for group, patterns in GROUP_PATTERNS:
        for pat in patterns:
            if pat in p:
                return group
    return None


def infer_big_five_group_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        host = urlparse(url).netloc.lower()
        # strip leading www.
        if host.startswith("www."):
            host = host[4:]
        for d, group in DOMAIN_TO_GROUP.items():
            if host.endswith(d):
                return group
        return None
    except Exception:
        return None


def infer_big_five_group(publisher_name: Optional[str], urls: List[str]) -> Optional[str]:
    # Try publisher string
    group = infer_big_five_group_from_publisher(publisher_name)
    if group:
        return group
    # Try any URL domains
    for u in urls:
        group2 = infer_big_five_group_from_url(u)
        if group2:
            return group2
    return None


def unique_nonempty(urls: List[Optional[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for u in urls:
        if isinstance(u, str):
            u2 = u.strip()
            if u2 and u2 not in seen:
                out.append(u2)
                seen.add(u2)
    return out


def ordinal(n: int) -> str:
    return "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class MemoirEntry(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None
    link: Optional[str] = None
    audiobook_link: Optional[str] = None
    additional_links: List[str] = Field(default_factory=list)

    # Optional helpful fields (if answer states them explicitly)
    hardcover_available: Optional[bool] = None
    audiobook_available: Optional[bool] = None
    audiobook_narrator: Optional[str] = None
    audiobook_author_narrated: Optional[bool] = None
    page_count_hardcover: Optional[str] = None
    genre: Optional[str] = None


class MemoirList(BaseModel):
    memoirs: List[MemoirEntry] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_memoirs() -> str:
    return """
Extract up to 8 candidate 2025 celebrity memoirs listed in the answer. For each memoir mentioned, extract the following fields exactly from the answer text (do not invent any):

- title: The full book title (string)
- author: The author's full name (string)
- publisher: The publisher or imprint name as written (string)
- publication_date: The stated publication or on-sale date (string, keep original format)
- link: A direct link to the publisher's official page for the book, or a reliable retailer page (URL as string). Prefer the publisher page if multiple are listed.
- audiobook_link: If an audiobook-specific link (e.g., Audible, Libro.fm, or publisher audiobook page) is explicitly provided in the answer, extract it. Otherwise null.
- additional_links: Any other URLs explicitly associated with this memoir in the answer (array of strings). Exclude duplicates of link or audiobook_link.

If the answer states availability information, also extract (else set to null):
- hardcover_available: true/false if explicitly mentioned
- audiobook_available: true/false if explicitly mentioned
- audiobook_narrator: narrator's name if explicitly mentioned
- audiobook_author_narrated: true/false if the answer explicitly confirms the author narrates the audiobook
- page_count_hardcover: the hardcover page count if given (string)
- genre: the genre label if provided (e.g., "memoir", "autobiography")

Return a JSON object with a single field:
{
  "memoirs": [ ... up to 8 MemoirEntry objects ... ]
}
If any field is missing in the answer for a memoir, set it to null (or [] for arrays).
Do not include any memoirs not explicitly mentioned in the answer.
"""


# -----------------------------------------------------------------------------
# Verification helpers for single memoir
# -----------------------------------------------------------------------------
async def verify_single_memoir(
    evaluator: Evaluator,
    parent_node,
    memoir: MemoirEntry,
    idx_zero_based: int,
) -> None:
    """
    Build verification sub-tree for one memoir and run verifications.
    Conforms to the rubric; some extra-rubric constraints treated as non-critical if inconsistent with the task.
    """
    i = idx_zero_based + 1
    node = evaluator.add_parallel(
        id=f"Memoir_{i}",
        desc=f"Evaluation of the {ordinal(i)} memoir and all its required attributes",
        parent=parent_node,
        critical=False,
    )

    # Gather sources
    sources = unique_nonempty([memoir.link, memoir.audiobook_link] + (memoir.additional_links or []))

    # Basic presence checks (critical)
    title_exists = evaluator.add_custom_node(
        result=bool(memoir.title and memoir.title.strip()),
        id=f"Memoir{i}_Title_Provided",
        desc=f"The response provides the book title for the {ordinal(i)} memoir",
        parent=node,
        critical=True,
    )
    author_exists = evaluator.add_custom_node(
        result=bool(memoir.author and memoir.author.strip()),
        id=f"Memoir{i}_Author_Provided",
        desc=f"The response provides the author's name for the {ordinal(i)} memoir",
        parent=node,
        critical=True,
    )
    publisher_exists = evaluator.add_custom_node(
        result=bool(memoir.publisher and memoir.publisher.strip()),
        id=f"Memoir{i}_Publisher_Provided",
        desc=f"The response provides the publisher name (including imprint if applicable) for the {ordinal(i)} memoir",
        parent=node,
        critical=True,
    )
    pubdate_exists = evaluator.add_custom_node(
        result=bool(memoir.publication_date and memoir.publication_date.strip()),
        id=f"Memoir{i}_Publication_Date_Provided",
        desc=f"The response provides the specific publication date for the {ordinal(i)} memoir",
        parent=node,
        critical=True,
    )
    link_exists = evaluator.add_custom_node(
        result=bool(sources),
        id=f"Memoir{i}_Link_Provided",
        desc=f"The response provides a direct link to the publisher/retailer page for the {ordinal(i)} memoir",
        parent=node,
        critical=True,
    )

    # Genre = memoir/autobiography (critical)
    genre_leaf = evaluator.add_leaf(
        id=f"Memoir{i}_Genre_Memoir",
        desc=f"The {ordinal(i)} book is a memoir or autobiography",
        parent=node,
        critical=True,
    )
    claim_genre = (
        f"The page(s) show that the book titled '{memoir.title or ''}' by {memoir.author or ''} "
        f"is a memoir or autobiography (a first-person narrative about the author's own life)."
    )
    await evaluator.verify(
        claim=claim_genre,
        node=genre_leaf,
        sources=sources,
        additional_instruction="Look for labels like 'memoir' or 'autobiography' in the book description, metadata, or category.",
    )

    # Celebrity/public figure status (critical)
    celeb_leaf = evaluator.add_leaf(
        id=f"Memoir{i}_Celebrity_Status",
        desc=f"The {ordinal(i)} memoir's author is a verifiable celebrity or public figure",
        parent=node,
        critical=True,
    )
    claim_celeb = (
        f"The author {memoir.author or ''} is an entertainer, athlete, media personality, politician, or otherwise a notable public figure."
    )
    await evaluator.verify(
        claim=claim_celeb,
        node=celeb_leaf,
        sources=sources,
        additional_instruction=(
            "Judge based on explicit hints on the provided page(s), e.g., 'actor', 'comedian', 'music artist', "
            "'athlete', 'TV host', 'influencer', 'journalist', 'politician'. If unclear or not supported, mark as not supported."
        ),
    )

    # Published by a Big Five or their imprint (critical)
    # First try deterministic inference; fallback to LLM simple verify when unknown.
    inferred_group = infer_big_five_group(memoir.publisher, sources)
    if inferred_group:
        evaluator.add_custom_node(
            result=True,
            id=f"Memoir{i}_Publisher_Big_Five",
            desc=f"The {ordinal(i)} memoir is published by a Big Five publisher or imprint ({inferred_group})",
            parent=node,
            critical=True,
        )
    else:
        pub_big5_leaf = evaluator.add_leaf(
            id=f"Memoir{i}_Publisher_Big_Five",
            desc=f"The {ordinal(i)} memoir is published by one of the Big Five publishers or their imprints",
            parent=node,
            critical=True,
        )
        claim_big5 = (
            f"The publisher/imprint '{memoir.publisher or ''}' belongs to one of the Big Five U.S. trade publishers "
            f"(Penguin Random House, HarperCollins, Simon & Schuster, Hachette Book Group, or Macmillan)."
        )
        await evaluator.verify(
            claim=claim_big5,
            node=pub_big5_leaf,
            sources=None,  # Allow general knowledge for imprint-to-parent mapping
            additional_instruction=(
                "Use your general industry knowledge of publishing imprints to decide membership. "
                "Do not require that the provided page explicitly states 'Big Five'."
            ),
        )

    # Published in 2025 (critical)
    pub2025_leaf = evaluator.add_leaf(
        id=f"Memoir{i}_Published_2025",
        desc=f"The {ordinal(i)} memoir was published in calendar year 2025",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The listed publication/on-sale date for this book is in 2025.",
        node=pub2025_leaf,
        sources=sources,
        additional_instruction="Check 'Publication Date', 'On Sale Date', or 'Release Date' fields. Accept if clearly in 2025.",
    )

    # Dual format availability (critical)
    dual_format_leaf = evaluator.add_leaf(
        id=f"Memoir{i}_Dual_Format",
        desc=f"The {ordinal(i)} memoir is available in both hardcover and audiobook formats",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Both hardcover and audiobook formats are available for this title.",
        node=dual_format_leaf,
        sources=sources,
        additional_instruction="Look for format listings or purchase options that explicitly include 'Hardcover' and 'Audiobook' (or audio download).",
    )

    # Author-narrated audiobook (critical)
    narrated_leaf = evaluator.add_leaf(
        id=f"Memoir{i}_Author_Narrated",
        desc=f"The audiobook of the {ordinal(i)} memoir is narrated by the author",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The audiobook for this title is narrated by the author {memoir.author or ''}.",
        node=narrated_leaf,
        sources=sources,
        additional_instruction="Look for 'narrated by' or 'read by the author'. A clear match between narrator and author is required.",
    )

    # Page count >= 250 (NOT in the original task; treat as non-critical to avoid unfair failure)
    pages_leaf = evaluator.add_leaf(
        id=f"Memoir{i}_Page_Count",
        desc=f"The {ordinal(i)} memoir hardcover edition has at least 250 pages",
        parent=node,
        critical=False,  # Relaxed due to mismatch with the original task requirements
    )
    await evaluator.verify(
        claim="The hardcover page count for this book is at least 250 pages.",
        node=pages_leaf,
        sources=sources,
        additional_instruction="If the page count is shown (e.g., '288 pages'), verify it's >= 250. If not shown, mark as not supported.",
    )

    # Verifiable publication metadata on the provided link (critical)
    verif_leaf = evaluator.add_leaf(
        id=f"Memoir{i}_Verifiable_Publication_Info",
        desc=f"The {ordinal(i)} memoir page confirms key publication details (publisher/date or ISBN)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The page confirms the book's publication metadata (publisher and publication date and/or an ISBN).",
        node=verif_leaf,
        sources=sources,
        additional_instruction="Confirm presence of publisher AND publication date, or of an ISBN, on the page.",
    )

    # Currently available for purchase as of March 2026 (NOT in the original task; treat as non-critical)
    avail_leaf = evaluator.add_leaf(
        id=f"Memoir{i}_Currently_Available",
        desc=f"The {ordinal(i)} memoir is currently available for purchase (not out of print) as of March 2026",
        parent=node,
        critical=False,  # Relaxed due to not being specified in the user's task
    )
    await evaluator.verify(
        claim="This book is currently available for purchase (not out of print).",
        node=avail_leaf,
        sources=sources,
        additional_instruction="Look for indicators like 'Add to cart', 'Buy Now', 'In Stock', or live retailer listings. If page clearly indicates unavailable/out-of-print, fail.",
    )


# -----------------------------------------------------------------------------
# Cross-memoir critical checks
# -----------------------------------------------------------------------------
def compute_publisher_groups(memoirs: List[MemoirEntry]) -> List[Optional[str]]:
    groups: List[Optional[str]] = []
    for m in memoirs:
        urls = unique_nonempty([m.link, m.audiobook_link] + (m.additional_links or []))
        groups.append(infer_big_five_group(m.publisher, urls))
    return groups


def compute_timeline_coverage(memoirs: List[MemoirEntry]) -> Tuple[bool, Dict]:
    """
    Return (ok, details) where ok means at least one Q1 2025 and at least one Q2-Q4 2025.
    """
    q1 = False
    q2_q4 = False
    details_list = []
    for idx, m in enumerate(memoirs):
        mon, yr = parse_month_year(m.publication_date)
        details_list.append({
            "index": idx + 1,
            "title": m.title,
            "publication_date": m.publication_date,
            "parsed_month": mon,
            "parsed_year": yr,
        })
        if yr == 2025 and mon is not None:
            if 1 <= mon <= 3:
                q1 = True
            elif 4 <= mon <= 12:
                q2_q4 = True
    return (q1 and q2_q4), {"parsed_dates": details_list, "has_q1": q1, "has_q2_q4": q2_q4}


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the 2025 celebrity memoirs task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Matches rubric: collection node is parallel
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

    # 1) Extract memoirs
    extracted: MemoirList = await evaluator.extract(
        prompt=prompt_extract_memoirs(),
        template_class=MemoirList,
        extraction_name="memoirs_extraction",
    )

    # Normalize to exactly 4 items (truncate or pad)
    items: List[MemoirEntry] = list(extracted.memoirs or [])
    if len(items) > 4:
        items = items[:4]
    while len(items) < 4:
        items.append(MemoirEntry())

    # Record a custom info about normalization
    evaluator.add_custom_info(
        {
            "total_listed_in_answer": len(extracted.memoirs or []),
            "evaluated_count": 4,
        },
        info_type="normalization",
        info_name="item_count_normalization",
    )

    # 2) Build subtrees for each memoir (parallel under root)
    # The rubric lists four "Memoir_i" blocks as NON-CRITICAL children under the collection node.
    for idx, m in enumerate(items):
        await verify_single_memoir(evaluator, root, m, idx)

    # 3) Cross-memoir critical checks
    # 3a) Publisher diversity: at least three distinct Big Five groups
    groups = compute_publisher_groups(items)
    distinct_groups = sorted({g for g in groups if g})
    diversity_ok = len(distinct_groups) >= 3
    evaluator.add_custom_node(
        result=diversity_ok,
        id="Publisher_Diversity",
        desc="The 4 memoirs collectively represent at least 3 different Big Five publishers",
        parent=root,
        critical=True,  # As rubric specifies
    )
    evaluator.add_custom_info(
        {"normalized_groups": groups, "distinct_groups": distinct_groups, "ok": diversity_ok},
        info_type="cross_check",
        info_name="publisher_diversity_details",
    )

    # 3b) Timeline coverage: one in Q1 2025 and one in Q2-Q4 2025
    timeline_ok, timeline_details = compute_timeline_coverage(items)
    evaluator.add_custom_node(
        result=timeline_ok,
        id="Timeline_Coverage",
        desc="The 4 memoirs include at least one published in Q1 (Jan–Mar 2025) and at least one in Q2–Q4 (Apr–Dec 2025)",
        parent=root,
        critical=True,  # As rubric specifies
    )
    evaluator.add_custom_info(
        timeline_details,
        info_type="cross_check",
        info_name="timeline_coverage_details",
    )

    return evaluator.get_summary()