import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# ------------------------------------------------------------------------------
# Task constants and helpers
# ------------------------------------------------------------------------------
TASK_ID = "translated_lit_awards_2024_2025"
TASK_DESCRIPTION = (
    "Identify books of translated literary fiction that won major English-language international literary awards "
    "(Pulitzer Prize for Fiction, Booker Prize, International Booker Prize, or National Book Award for Translated Literature) "
    "with winner announcements made between October 2024 and June 2025. At least one of the identified books must have a "
    "documented 'first-time' milestone for its award category (e.g., first short story collection to win the specific prize, "
    "or first translation from a particular language to win). For each identified book, provide comprehensive documentation: "
    "1) Award information (award name, winner announcement date, category acceptance for translations/short stories where applicable, "
    "and an official/credible announcement URL). 2) Publication details (publisher name and type (major or independent), English publication "
    "date (Jan 2024–Dec 2025), valid ISBN, format and page count, and a publisher or major retailer URL). 3) Translation information "
    "(source language, translator name, evidence of translator expertise/recognition, and a URL confirming translation details). "
    "4) First-time achievement (if applicable): clear description, supporting evidence, and a reference URL."
)

ALLOWED_AWARDS = {
    "pulitzer prize for fiction",
    "booker prize",
    "international booker prize",
    "national book award for translated literature",
    "national book awards: translated literature",
}

ANNOUNCEMENT_START = datetime(2024, 10, 1)
ANNOUNCEMENT_END = datetime(2025, 6, 30)
ENGLISH_PUB_START = datetime(2024, 1, 1)
ENGLISH_PUB_END = datetime(2025, 12, 31)
ELIGIBILITY_START_DATE = datetime(2024, 10, 1)  # As implied by rubric requirement


# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------
def normalize_text(s: Optional[str]) -> str:
    return (s or "").strip()

def normalize_award_name(name: Optional[str]) -> str:
    n = normalize_text(name).lower()
    # Common normalizations
    n = n.replace("–", "-")
    n = n.replace("’", "'")
    n = n.replace("  ", " ")
    # Some synonyms/variants handling
    if "pulitzer" in n and "fiction" in n:
        return "pulitzer prize for fiction"
    if "international booker" in n:
        return "international booker prize"
    if "booker prize" in n and "international" not in n:
        return "booker prize"
    if "national book award" in n and "translated" in n:
        return "national book award for translated literature"
    if "translated literature" in n and "national book awards" in n:
        return "national book awards: translated literature"
    return n

def award_in_allowed_set(name: Optional[str]) -> bool:
    n = normalize_award_name(name)
    if not n:
        return False
    # fuzzy match: allow tokens
    if n in ALLOWED_AWARDS:
        return True
    # fallback token checks
    if "pulitzer" in n and "fiction" in n:
        return True
    if n == "booker prize" or ("booker" in n and "prize" in n and "international" not in n):
        return True
    if "international" in n and "booker" in n:
        return True
    if "national book award" in n and "translated" in n:
        return True
    return False

def parse_date_flexible(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    s = normalize_text(date_str)
    # Try multiple formats
    fmts = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %b %Y",
        "%d %B %Y",
        "%Y-%m",
        "%Y/%m",
        "%Y",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            # If only year or year-month, coerce to start of period
            if fmt == "%Y":
                return datetime(int(s), 1, 1)
            if fmt in ("%Y-%m", "%Y/%m"):
                parts = re.split(r"[-/]", s)
                return datetime(int(parts[0]), int(parts[1]), 1)
            return dt
        except Exception:
            continue
    # Try extracting an ISO date substring
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            pass
    # Try Month YYYY
    m2 = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", s, re.IGNORECASE)
    if m2:
        try:
            return datetime.strptime(m2.group(0), "%B %Y")
        except Exception:
            pass
    return None

def date_in_window(dt: Optional[datetime], start: datetime, end: datetime) -> bool:
    if not dt:
        return False
    return start <= dt <= end

def isbn_is_valid_simple(isbn: Optional[str]) -> bool:
    if not isbn:
        return False
    s = re.sub(r"[^0-9Xx]", "", isbn)
    if len(s) == 10:
        # Basic ISBN-10 checksum
        total = 0
        for i, ch in enumerate(s):
            if ch in "Xx":
                val = 10
            elif ch.isdigit():
                val = int(ch)
            else:
                return False
            total += val * (10 - i)
        return total % 11 == 0
    if len(s) == 13 and s.isdigit():
        # Basic ISBN-13 checksum
        total = 0
        for i, ch in enumerate(s):
            val = int(ch)
            total += val * (1 if i % 2 == 0 else 3)
        return total % 10 == 0
    # If not checksum-valid, still allow if it looks like 10 or 13 digits (to be lenient with extraction)
    if len(s) in (10, 13):
        return True
    return False

def format_is_provided(fmt: Optional[str]) -> bool:
    if not fmt:
        return False
    f = normalize_text(fmt).lower()
    return any(x in f for x in ["hardcover", "paperback", "trade paperback", "hb", "pb"])

def looks_like_major_or_independent(value: Optional[str]) -> bool:
    if not value:
        return False
    v = normalize_text(value).lower()
    return v in ("major", "independent", "indie", "major publisher", "independent publisher")

def is_international_booker(name: Optional[str]) -> bool:
    return "international booker" in normalize_award_name(name)

def non_english_language(lang: Optional[str]) -> bool:
    if not lang:
        return False
    return normalize_text(lang).lower() not in ["english", "en", "eng"]


# ------------------------------------------------------------------------------
# Data models for extraction
# ------------------------------------------------------------------------------
class AwardInfo(BaseModel):
    name: Optional[str] = None
    announcement_date: Optional[str] = None  # keep as string; we'll parse
    announcement_url: Optional[str] = None
    category_or_notes: Optional[str] = None
    category_rules_url: Optional[str] = None

class PublicationInfo(BaseModel):
    english_publication_date: Optional[str] = None
    publisher_name: Optional[str] = None
    publisher_type: Optional[str] = None  # "major" or "independent"
    publisher_type_justification: Optional[str] = None
    publication_url: Optional[str] = None  # publisher or major retailer
    publisher_info_url: Optional[str] = None  # optional second source about publisher classification
    isbn: Optional[str] = None
    book_format: Optional[str] = None
    page_count: Optional[str] = None
    intl_booker_uk_ie_publication_url: Optional[str] = None  # used only if award is Intl Booker

class TranslationInfo(BaseModel):
    author_name: Optional[str] = None
    is_translation: Optional[str] = None  # "yes"/"no"
    source_language: Optional[str] = None
    translator_name: Optional[str] = None
    translation_reference_url: Optional[str] = None  # confirms language/translator
    translator_expertise_evidence_text: Optional[str] = None
    translator_expertise_url: Optional[str] = None
    alive_evidence_text: Optional[str] = None
    alive_evidence_url: Optional[str] = None

class MilestoneInfo(BaseModel):
    description: Optional[str] = None
    evidence_text: Optional[str] = None
    url: Optional[str] = None

class BookItem(BaseModel):
    title: Optional[str] = None
    genre: Optional[str] = None  # "novel", "short story collection", etc.
    genre_url: Optional[str] = None
    award: AwardInfo = Field(default_factory=AwardInfo)
    publication: PublicationInfo = Field(default_factory=PublicationInfo)
    translation: TranslationInfo = Field(default_factory=TranslationInfo)
    first_time: Optional[MilestoneInfo] = None

class BooksExtraction(BaseModel):
    books: List[BookItem] = Field(default_factory=list)


# ------------------------------------------------------------------------------
# Extraction prompt
# ------------------------------------------------------------------------------
def prompt_extract_books() -> str:
    return """
    Extract up to 2 books from the answer that the author claims meet the task requirements (translated literary fiction winning one of the specified awards).
    For each book, extract the following fields exactly as stated in the answer (do not invent):
    - title
    - genre (e.g., "novel", "short story collection", "literary fiction"; be specific if stated)
    - genre_url (a URL supporting the genre classification; can be publisher page or credible source)
    - award:
        - name (e.g., "International Booker Prize", "Pulitzer Prize for Fiction", "Booker Prize", or "National Book Award for Translated Literature")
        - announcement_date (as written; any reasonable date format)
        - announcement_url (URL to official announcement or a credible source confirming the winner and the date)
        - category_or_notes (any notes describing category acceptance for translations or short story collections)
        - category_rules_url (URL to rules/category acceptance if provided)
    - publication:
        - english_publication_date (as written)
        - publisher_name
        - publisher_type (use "major" or "independent" if stated; otherwise leave null)
        - publisher_type_justification (short text stating why "major" or "independent" classification is appropriate, if provided)
        - publication_url (URL to publisher page or a major retailer with publication details)
        - publisher_info_url (optional: another URL supporting the classification of the publisher)
        - isbn (as provided; include hyphens if present)
        - book_format (e.g., hardcover/paperback; can include multiple if listed)
        - page_count (as a number or text, exactly as in the answer)
        - intl_booker_uk_ie_publication_url (URL that shows UK/Ireland publication evidence; only if award is the International Booker)
    - translation:
        - author_name
        - is_translation (use "yes" if the work is translated into English; otherwise "no" or null)
        - source_language (the original language; do not use "English")
        - translator_name
        - translation_reference_url (URL confirming translation details)
        - translator_expertise_evidence_text (short text about awards/prior recognition or bio)
        - translator_expertise_url (URL to bio/recognition evidence)
        - alive_evidence_text (text asserting author and translator were alive as of the eligibility start)
        - alive_evidence_url (URL supporting the alive claim(s))
    - first_time (only if a first-time milestone is claimed for the book):
        - description (clear statement of the milestone, e.g., first short story collection to win X)
        - evidence_text (short supporting text)
        - url (URL documenting the milestone claim)

    Rules:
    - Extract only what is present in the answer text. If something is not present, return null for that field.
    - If multiple books are present, choose the first two that appear relevant to the task.
    - Ensure URLs are full and valid; if a URL is missing protocol, prepend http://
    """


# ------------------------------------------------------------------------------
# Verification helpers per book
# ------------------------------------------------------------------------------
async def verify_award_information(
    evaluator: Evaluator,
    parent,
    book: BookItem,
    idx: int
):
    node = evaluator.add_parallel(
        id=f"book_{idx}_award_information",
        desc=f"Book #{idx + 1}: Award information is provided and satisfies award-related constraints.",
        parent=parent,
        critical=True
    )

    # 1) Award in allowed set
    in_allowed = award_in_allowed_set(book.award.name)
    evaluator.add_custom_node(
        result=in_allowed,
        id=f"book_{idx}_award_in_allowed_set",
        desc="Award is one of: Pulitzer Prize for Fiction, Booker Prize, International Booker Prize, or National Book Award for Translated Literature (or equivalent named category).",
        parent=node,
        critical=True
    )

    # 2) Winner announcement date in window
    announced_dt = parse_date_flexible(book.award.announcement_date)
    date_ok = date_in_window(announced_dt, ANNOUNCEMENT_START, ANNOUNCEMENT_END)
    evaluator.add_custom_node(
        result=date_ok,
        id=f"book_{idx}_winner_announcement_date_in_window",
        desc="Winner announcement date is provided and falls between October 2024 and June 2025 (inclusive).",
        parent=node,
        critical=True
    )

    # 3) Category acceptance / fiction fit (verify with provided URLs)
    fit_leaf = evaluator.add_leaf(
        id=f"book_{idx}_award_category_translation_or_fiction_fit",
        desc="Award category is appropriate for translated literature and/or general fiction as applicable, and (if the work is a short story collection) the award accepts short story collections.",
        parent=node,
        critical=True
    )
    book_genre = normalize_text(book.genre).lower()
    is_short_story = "short" in book_genre and "story" in book_genre
    claim = (
        f"The award '{book.award.name}' and its category accept translated literature and are appropriate for this book's classification "
        f"('{book.genre}'). Additionally, it {'does' if is_short_story else 'does not necessarily'} accept short story collections as needed for this book."
    )
    sources = [u for u in [book.award.category_rules_url, book.award.announcement_url, book.genre_url] if u]
    await evaluator.verify(
        claim=claim,
        node=fit_leaf,
        sources=sources,
        additional_instruction=(
            "Check the award rules or credible sources to confirm: (a) the award/category allows translated works; "
            "(b) if the book is a short story collection, that the award accepts such collections."
        )
    )

    # 4) Award announcement URL verifies win and date
    ann_leaf = evaluator.add_leaf(
        id=f"book_{idx}_award_announcement_url",
        desc="A URL to an official award announcement or other credible source is provided to verify the win and announcement date.",
        parent=node,
        critical=True
    )
    claim2 = (
        f"This webpage is the official or a credible announcement stating that '{book.title}' won the '{book.award.name}', "
        f"with the winner announcement on or around {book.award.announcement_date}."
    )
    await evaluator.verify(
        claim=claim2,
        node=ann_leaf,
        sources=book.award.announcement_url,
        additional_instruction="Confirm both the win and the approximate announcement date."
    )

    return node


async def verify_publication_details(
    evaluator: Evaluator,
    parent,
    book: BookItem,
    idx: int
):
    node = evaluator.add_parallel(
        id=f"book_{idx}_publication_details",
        desc=f"Book #{idx + 1}: Publication details are provided and satisfy publication-related constraints.",
        parent=parent,
        critical=True
    )

    # 1) English publication date in window
    pub_dt = parse_date_flexible(book.publication.english_publication_date)
    pub_ok = date_in_window(pub_dt, ENGLISH_PUB_START, ENGLISH_PUB_END)
    evaluator.add_custom_node(
        result=pub_ok,
        id=f"book_{idx}_english_publication_date_in_window",
        desc="English-language publication date is provided and falls between January 2024 and December 2025 (inclusive).",
        parent=node,
        critical=True
    )

    # 2) Publisher name provided
    evaluator.add_custom_node(
        result=bool(normalize_text(book.publication.publisher_name)),
        id=f"book_{idx}_publisher_name_provided",
        desc="Publisher name is provided.",
        parent=node,
        critical=True
    )

    # 3) Publisher type and justification (verify with URLs)
    pub_type_leaf = evaluator.add_leaf(
        id=f"book_{idx}_publisher_type_and_justification",
        desc="Publisher is categorized as major publishing house or established independent publisher, with justification/evidence.",
        parent=node,
        critical=True
    )
    claim = (
        f"The publisher '{book.publication.publisher_name}' is appropriately classified as '{book.publication.publisher_type}', "
        f"supported by the provided justification: {book.publication.publisher_type_justification}."
    )
    sources = [u for u in [book.publication.publication_url, book.publication.publisher_info_url] if u]
    await evaluator.verify(
        claim=claim,
        node=pub_type_leaf,
        sources=sources,
        additional_instruction=(
            "Judge whether the classification ('major' or 'independent') is reasonably supported by the given sources "
            "(e.g., scale, imprint ownership, reputation). If not enough evidence, mark as unsupported."
        )
    )

    # 4) International Booker UK/IE publication requirement (only applicable if award is International Booker)
    if is_international_booker(book.award.name):
        intl_leaf = evaluator.add_leaf(
            id=f"book_{idx}_intl_booker_uk_ie_publication_if_applicable",
            desc="For International Booker Prize: evidence is provided that the English translation was published in the UK or Ireland.",
            parent=node,
            critical=True
        )
        claim_intl = (
            "This page shows that the English translation was published in the UK or Ireland (as required for the International Booker Prize)."
        )
        await evaluator.verify(
            claim=claim_intl,
            node=intl_leaf,
            sources=book.publication.intl_booker_uk_ie_publication_url or book.publication.publication_url,
            additional_instruction="Look for explicit mention of UK or Irish publication; accept publisher pages or credible retailer listings."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"book_{idx}_intl_booker_uk_ie_publication_not_applicable",
            desc="International Booker UK/IE publication requirement: Not applicable (award is not the International Booker Prize).",
            parent=node,
            critical=True
        )

    # 5) Valid ISBN provided (simple validation)
    evaluator.add_custom_node(
        result=isbn_is_valid_simple(book.publication.isbn),
        id=f"book_{idx}_valid_isbn_provided",
        desc="A valid ISBN is provided.",
        parent=node,
        critical=True
    )

    # 6) Physical format provided
    evaluator.add_custom_node(
        result=format_is_provided(book.publication.book_format),
        id=f"book_{idx}_physical_format_provided",
        desc="Physical book format is provided (hardcover and/or paperback).",
        parent=node,
        critical=True
    )

    # 7) Page count verifiable
    page_leaf = evaluator.add_leaf(
        id=f"book_{idx}_page_count_verifiable",
        desc="Page count is provided and is verifiable from the cited publisher page or a major book retailer.",
        parent=node,
        critical=True
    )
    claim_pg = f"The page shows that the page count for '{book.title}' is '{book.publication.page_count}'."
    await evaluator.verify(
        claim=claim_pg,
        node=page_leaf,
        sources=book.publication.publication_url,
        additional_instruction="Verify the stated page count on the page."
    )

    # 8) Publication reference URL credibility
    ref_leaf = evaluator.add_leaf(
        id=f"book_{idx}_publication_reference_url",
        desc="A URL to the publisher page or a major book retailer is provided to verify publication details.",
        parent=node,
        critical=True
    )
    claim_ref = "This page is a publisher or major retailer listing for the book and can be used to verify publication details (date/ISBN/format/page count)."
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=book.publication.publication_url,
        additional_instruction="Assess credibility and relevance of the page to verify publication details."
    )

    return node


async def verify_translation_information(
    evaluator: Evaluator,
    parent,
    book: BookItem,
    idx: int
):
    node = evaluator.add_parallel(
        id=f"book_{idx}_translation_information",
        desc=f"Book #{idx + 1}: Translation details are provided and satisfy translation-related constraints.",
        parent=parent,
        critical=True
    )

    # 1) Work is translation into English (verify by URL)
    trans_leaf = evaluator.add_leaf(
        id=f"book_{idx}_work_is_translation_into_english",
        desc="The book is identified as a work translated into English.",
        parent=node,
        critical=True
    )
    claim_trans = f"The book '{book.title}' is translated into English."
    await evaluator.verify(
        claim=claim_trans,
        node=trans_leaf,
        sources=book.translation.translation_reference_url,
        additional_instruction="Look for explicit mention of translation into English on the source page."
    )

    # 2) Source language not English (custom)
    evaluator.add_custom_node(
        result=non_english_language(book.translation.source_language),
        id=f"book_{idx}_source_language_not_english",
        desc="The original/source language is provided and is not English.",
        parent=node,
        critical=True
    )

    # 3) Translator name provided (custom)
    evaluator.add_custom_node(
        result=bool(normalize_text(book.translation.translator_name)),
        id=f"book_{idx}_translator_name_provided",
        desc="Translator name is provided.",
        parent=node,
        critical=True
    )

    # 4) Translator expertise evidence (verify by URL)
    exp_leaf = evaluator.add_leaf(
        id=f"book_{idx}_translator_expertise_evidence",
        desc="Evidence of translator expertise/recognition is provided (awards, notable translations, credible bio, etc.).",
        parent=node,
        critical=True
    )
    claim_exp = (
        f"The translator '{book.translation.translator_name}' has recognized expertise or credentials: "
        f"{book.translation.translator_expertise_evidence_text}."
    )
    await evaluator.verify(
        claim=claim_exp,
        node=exp_leaf,
        sources=book.translation.translator_expertise_url or book.translation.translation_reference_url,
        additional_instruction="Accept credible sources like publisher bios, award pages, or reputable profiles."
    )

    # 5) Author and translator alive at eligibility start (verify by URL)
    alive_leaf = evaluator.add_leaf(
        id=f"book_{idx}_author_translator_alive_at_eligibility_start",
        desc="Evidence is provided that both author and translator were alive at the beginning of the award eligibility period.",
        parent=node,
        critical=True
    )
    claim_alive = (
        f"As of {ELIGIBILITY_START_DATE.strftime('%Y-%m-%d')}, both the author '{book.translation.author_name}' "
        f"and translator '{book.translation.translator_name}' were alive."
    )
    await evaluator.verify(
        claim=claim_alive,
        node=alive_leaf,
        sources=book.translation.alive_evidence_url,
        additional_instruction="Look for credible bios or news entries indicating life status around the relevant date."
    )

    # 6) Translation reference URL confirms language and translator
    trans_ref_leaf = evaluator.add_leaf(
        id=f"book_{idx}_translation_reference_url",
        desc="A URL is provided confirming translation details (source language and translator).",
        parent=node,
        critical=True
    )
    claim_ref = (
        f"This page confirms that the original/source language is '{book.translation.source_language}' "
        f"and that the translator is '{book.translation.translator_name}'."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=trans_ref_leaf,
        sources=book.translation.translation_reference_url,
        additional_instruction="Confirm both translator identity and the non-English source language."
    )

    return node


async def verify_genre_requirement(
    evaluator: Evaluator,
    parent,
    book: BookItem,
    idx: int
):
    # Verify literary fiction classification (novel or short story collection)
    leaf = evaluator.add_leaf(
        id=f"book_{idx}_genre_requirement",
        desc="The book is classified as literary fiction (novel or short story collection).",
        parent=parent,
        critical=True
    )
    claim = (
        f"The book '{book.title}' is a work of literary fiction classified as '{book.genre}' "
        f"(novel or short story collection)."
    )
    sources = [u for u in [book.genre_url, book.publication.publication_url] if u]
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm classification; if 'short story collection', ensure it is described as such."
    )
    return leaf


async def verify_milestone_optional(
    evaluator: Evaluator,
    parent_optional_container,
    book: BookItem,
    idx: int
):
    # Place optional milestone details under a non-critical container outside critical path
    node = evaluator.add_parallel(
        id=f"book_{idx}_first_time_milestone_details",
        desc=f"Book #{idx + 1}: First-time milestone (if claimed) is described and supported.",
        parent=parent_optional_container,
        critical=False
    )
    # Description provided
    evaluator.add_custom_node(
        result=bool(book.first_time and normalize_text(book.first_time.description)),
        id=f"book_{idx}_milestone_description",
        desc="A clear description of the first-time milestone is provided (when claimed).",
        parent=node,
        critical=False
    )
    # Evidence supported by URL
    evidence_leaf = evaluator.add_leaf(
        id=f"book_{idx}_milestone_evidence",
        desc="Evidence supporting the milestone claim is provided (when claimed).",
        parent=node,
        critical=False
    )
    claim_ev = (
        f"This page documents the claimed first-time milestone for '{book.title}': "
        f"{book.first_time.description if book.first_time else ''}"
    )
    await evaluator.verify(
        claim=claim_ev,
        node=evidence_leaf,
        sources=(book.first_time.url if book.first_time else None),
        additional_instruction="Evaluate whether the page explicitly states or credibly supports the claimed 'first-time' milestone."
    )
    # Reference URL credibility
    url_leaf = evaluator.add_leaf(
        id=f"book_{idx}_milestone_reference_url",
        desc="A URL documenting the milestone claim is provided (when claimed).",
        parent=node,
        critical=False
    )
    claim_url = "This is an official announcement or credible source page documenting the milestone claim."
    await evaluator.verify(
        claim=claim_url,
        node=url_leaf,
        sources=(book.first_time.url if book.first_time else None),
        additional_instruction="Assess credibility; accept official prize site, respected news, or award authority sources."
    )
    return node


# ------------------------------------------------------------------------------
# Book verification orchestrator
# ------------------------------------------------------------------------------
async def verify_book_core(
    evaluator: Evaluator,
    parent_core_node,  # critical container
    book: BookItem,
    idx: int
):
    # Basic presence check for book identification
    evaluator.add_custom_node(
        result=bool(normalize_text(book.title) and normalize_text(book.award.name)),
        id=f"book_{idx}_identified",
        desc=f"Book #{idx + 1} is identified with a title and award name.",
        parent=parent_core_node,
        critical=True
    )

    # Award information
    await verify_award_information(evaluator, parent_core_node, book, idx)
    # Publication details
    await verify_publication_details(evaluator, parent_core_node, book, idx)
    # Translation information
    await verify_translation_information(evaluator, parent_core_node, book, idx)
    # Genre requirement
    await verify_genre_requirement(evaluator, parent_core_node, book, idx)


# ------------------------------------------------------------------------------
# Main evaluation entry point
# ------------------------------------------------------------------------------
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

    # Record GT/context info
    evaluator.add_ground_truth({
        "allowed_awards": sorted(list(ALLOWED_AWARDS)),
        "announcement_window": {
            "start": ANNOUNCEMENT_START.strftime("%Y-%m-%d"),
            "end": ANNOUNCEMENT_END.strftime("%Y-%m-%d"),
        },
        "english_publication_window": {
            "start": ENGLISH_PUB_START.strftime("%Y-%m-%d"),
            "end": ENGLISH_PUB_END.strftime("%Y-%m-%d"),
        },
        "eligibility_alive_date": ELIGIBILITY_START_DATE.strftime("%Y-%m-%d"),
        "requirement": "At least one identified book must include a documented first-time milestone."
    })

    # Extract books from the answer
    extracted: BooksExtraction = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    # Keep only first 2 books (as optional second)
    books: List[BookItem] = (extracted.books or [])[:2]
    # Ensure at least one placeholder if none
    if not books:
        books = [BookItem()]

    # Build the evaluation tree structure
    # Critical overall checks container
    overall = evaluator.add_parallel(
        id="overall_checks",
        desc="Overall required checks: structured presentation, at least one fully documented eligible book, and milestone across books.",
        parent=root,
        critical=True
    )

    # 1) Response presentation structure and sources
    pres_leaf = evaluator.add_leaf(
        id="response_presentation",
        desc="Findings are presented in a structured format and include source URLs for verification of required claims.",
        parent=overall,
        critical=True
    )
    await evaluator.verify(
        claim="The answer is well-structured (e.g., sections/fields per book) and provides source URLs for each required claim.",
        node=pres_leaf,
        additional_instruction="Look for a clear, structured layout and presence of URLs for award announcement, publication details, translation info, and milestone."
    )

    # 2) Book #1 core (critical)
    book1_core = evaluator.add_parallel(
        id="book_1_core",
        desc="Book #1: At least one eligible book is identified and fully documented (required).",
        parent=overall,
        critical=True
    )
    await verify_book_core(evaluator, book1_core, books[0], 0)

    # Optional container (non-critical): holds Book #2 core and milestone details for any book
    optional_container = evaluator.add_parallel(
        id="optional_container",
        desc="Optional items: Book #2 core (if provided) and milestone details per book.",
        parent=root,  # Keep outside critical path due to framework constraint
        critical=False
    )

    # 3) Book #2 core (optional)
    if len(books) > 1:
        book2_core = evaluator.add_parallel(
            id="book_2_core",
            desc="Book #2: Optional additional book evaluated with the same criteria.",
            parent=optional_container,
            critical=False
        )
        await verify_book_core(evaluator, book2_core, books[1], 1)

    # 4) Per-book optional milestone details (non-critical)
    # Book 1
    await verify_milestone_optional(evaluator, optional_container, books[0], 0)
    # Book 2 (if present)
    if len(books) > 1:
        await verify_milestone_optional(evaluator, optional_container, books[1], 1)

    # 5) Cross-book critical requirement: at least one documented first-time milestone
    # Determine pass/fail using the created milestone nodes (description + evidence + reference_url)
    def milestone_pass_for_book(i: int) -> bool:
        desc_node = evaluator.find_node(f"book_{i}_milestone_description")
        ev_node = evaluator.find_node(f"book_{i}_milestone_evidence")
        url_node = evaluator.find_node(f"book_{i}_milestone_reference_url")
        # Consider milestone satisfied if description provided and both evidence+reference url verifications passed
        return (
            (desc_node is not None and desc_node.status == "passed") and
            (ev_node is not None and ev_node.status == "passed") and
            (url_node is not None and url_node.status == "passed")
        )

    has_milestone = milestone_pass_for_book(0) or (len(books) > 1 and milestone_pass_for_book(1))

    evaluator.add_custom_node(
        result=has_milestone,
        id="at_least_one_first_time_milestone_across_books",
        desc="At least one identified book includes a documented first-time milestone for its award category, with supporting evidence and a reference URL.",
        parent=overall,
        critical=True
    )

    # Return summary
    return evaluator.get_summary()