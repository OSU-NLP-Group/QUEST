import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "award_winning_fiction_2024_2025"
TASK_DESCRIPTION = (
    "Identify a fiction novel that meets all of the following criteria: "
    "The book won a major literary award (Pulitzer Prize for Fiction, National Book Award for Fiction, or Booker Prize) in 2024 or 2025. "
    "The book was published between 2023 and 2024 (inclusive). "
    "The book's author holds at least a master's degree from an accredited university. "
    "The book contains between 100 and 350 pages. "
    "The book was published by a major traditional publishing house. "
    "The book is available in English (either originally written in English or professionally translated into English). "
    "Provide the title of the book, the author's name, the specific award(s) won, the year of publication, the number of pages, "
    "the publisher, and the author's highest degree with the institution name."
)

ALLOWED_AWARDS = [
    "Pulitzer Prize for Fiction",
    "National Book Award for Fiction",
    "Booker Prize",
]

PUB_YEAR_MIN = 2023
PUB_YEAR_MAX = 2024
MIN_PAGES = 100
MAX_PAGES = 350


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AwardInfo(BaseModel):
    name: Optional[str] = None  # e.g., "Booker Prize"
    year: Optional[str] = None  # e.g., "2024"
    urls: List[str] = Field(default_factory=list)  # explicit URLs from the answer only


class BookExtraction(BaseModel):
    # Core identification
    title: Optional[str] = None
    author_name: Optional[str] = None

    # Awards (list as presented in the answer; do not invent)
    awards: List[AwardInfo] = Field(default_factory=list)

    # Publication year
    publication_year: Optional[str] = None        # four-digit string if available
    publication_year_number: Optional[int] = None # integer if clearly provided
    publication_year_urls: List[str] = Field(default_factory=list)

    # Page count
    page_count: Optional[str] = None              # e.g., "320 pages"
    page_count_number: Optional[int] = None       # integer if clearly provided
    page_count_urls: List[str] = Field(default_factory=list)

    # Publisher
    publisher: Optional[str] = None
    publisher_urls: List[str] = Field(default_factory=list)         # publisher's book page, etc.
    publisher_status_urls: List[str] = Field(default_factory=list)  # sources showing "major traditional publisher" status

    # Language availability
    language: Optional[str] = None
    language_urls: List[str] = Field(default_factory=list)

    # Genre (if explicitly stated)
    genre_label: Optional[str] = None

    # Author education
    author_degree: Optional[str] = None          # e.g., "MA in English"
    author_institution: Optional[str] = None     # e.g., "Harvard University"
    author_education_urls: List[str] = Field(default_factory=list)  # biography page, university page, etc.


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_book() -> str:
    return """
Extract the single book (one fiction novel) that the answer is presenting as satisfying all constraints. Do not invent or infer new information; extract only what is explicitly present in the answer. Return the following JSON fields:

- title: the book title exactly as written in the answer; null if missing.
- author_name: the author's full name; null if missing.

- awards: an array. Include every award that the answer explicitly claims the book has WON (not just nominated/longlisted/shortlisted). For each award object:
  - name: the award name exactly as written (e.g., "Booker Prize", "Pulitzer Prize for Fiction", "National Book Award for Fiction"); null if missing.
  - year: the year for that specific award as written (prefer a 4-digit string like "2024"); null if missing.
  - urls: a list of explicit URLs cited in the answer that support the "won" claim for this award. Extract only URLs explicitly present in the answer. If none are present, return an empty list.

- publication_year: the book's publication year for the relevant/main edition referenced by the answer, as a 4-digit string if available; else null.
- publication_year_number: the same publication year as an integer if clearly provided; else null.
- publication_year_urls: list of explicit URLs in the answer that support the publication year; empty list if none.

- page_count: the page count text as written (e.g., "320 pages" or "320"); null if missing.
- page_count_number: the numeric page count if clearly a single number; else null.
- page_count_urls: list of explicit URLs in the answer supporting the page count; empty list if none.

- publisher: the publisher/imprint name as written (e.g., "Penguin Random House", "Knopf"); null if missing.
- publisher_urls: list of explicit URLs in the answer showing that this publisher/imprint published the book; empty list if none.
- publisher_status_urls: list of explicit URLs in the answer supporting that the publisher is a major traditional publishing house (e.g., Big Five or equivalent; Wikipedia or credible sources). If not provided, return an empty list.

- language: the language availability claim as written that indicates English availability (e.g., "English", "Available in English", "Translated into English"); null if missing.
- language_urls: list of explicit URLs in the answer supporting the English availability; empty list if none.

- genre_label: any explicit genre label from the answer that indicates "fiction" or "novel"; null if missing.

- author_degree: the highest degree of the author as written (e.g., "MA", "MFA", "PhD", "JD"); null if missing.
- author_institution: the institution awarding that degree as written; null if missing.
- author_education_urls: list of explicit URLs in the answer supporting the degree and institution; empty list if none.

IMPORTANT:
- Extract only URLs explicitly present in the answer text. Do not invent or infer URLs.
- For each 'urls' field, include all URLs cited for that item in the answer (plain URLs or in markdown).
- If something is missing from the answer, set it to null (or empty list for URLs arrays).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def first_non_empty_str(*vals: Optional[str]) -> str:
    for v in vals:
        if v and str(v).strip():
            return str(v).strip()
    return ""

def safe_int(val: Optional[str | int]) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    s = str(val)
    m = re.search(r"\b(\d{4})\b", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    try:
        return int(re.sub(r"[^\d]", "", s))
    except Exception:
        return None

def parse_page_int(text: Optional[str], fallback: Optional[int]) -> Optional[int]:
    if isinstance(fallback, int):
        return fallback
    if not text:
        return None
    # Prefer a single integer number from the text
    m = re.search(r"\b(\d{2,4})\b", text.replace(",", ""))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def combine_sources(*lists: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out

def pick_award_for_checks(awards: List[AwardInfo]) -> Optional[AwardInfo]:
    """
    Prefer the first award that looks like one of the allowed awards and has a 2024/2025 year;
    otherwise fall back to the first award with both name and year; otherwise None.
    """
    allowed_keys = [
        "pulitzer prize for fiction",
        "national book award for fiction",
        "booker prize",
    ]
    candidates: List[AwardInfo] = []
    for aw in awards or []:
        if aw and (aw.name or aw.year):
            candidates.append(aw)

    # Priority 1: allowed name + allowed year
    for aw in candidates:
        n = (aw.name or "").lower()
        y = safe_int(aw.year)
        if any(k in n for k in allowed_keys) and (y in (2024, 2025)):
            return aw

    # Priority 2: allowed name (even if year off)
    for aw in candidates:
        n = (aw.name or "").lower()
        if any(k in n for k in allowed_keys) and (aw.year is not None):
            return aw

    # Priority 3: first reasonably complete award with name and year
    for aw in candidates:
        if (aw.name and aw.year):
            return aw

    # Priority 4: anything with a name at least
    for aw in candidates:
        if aw.name:
            return aw

    return None

def is_degree_string_at_least_masters(degree: Optional[str]) -> bool:
    if not degree:
        return False
    s = degree.lower()
    # Common master's/professional/doctoral degree tokens
    master_tokens = [
        "master", "m.a.", "ma ", "mfa", "m.f.a", "msc", "m.sc", "ms ", "m.s.", "mphil", "m.phil",
        "meng", "m.eng", "mba", "m.b.a", "mpp", "m.p.p", "mrp", "m.r.p", "mlis", "m.l.i.s", "m.ed", "m.ed.", "med "
    ]
    professional_or_doctoral = [
        "phd", "ph.d", "dphil", "d.phil", "jd", "j.d", "md", "m.d", "edd", "ed.d", "sci.d", "scd", "dr", "doctor of"
    ]
    if any(tok in s for tok in professional_or_doctoral):
        return True
    if any(tok in s for tok in master_tokens):
        return True
    # also match patterns like "MSc", "MA", "MS" exactly (case-insensitive)
    exacts = [r"\bmsc\b", r"\bms\b", r"\bma\b", r"\bmfa\b", r"\bmeng\b", r"\bmphil\b", r"\bmba\b", r"\bmed\b", r"\bmlis\b"]
    for pat in exacts:
        if re.search(pat, s, flags=re.IGNORECASE):
            return True
    return False


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_award_information(evaluator: Evaluator, parent, info: BookExtraction):
    node = evaluator.add_sequential(
        id="Award_Information",
        desc="The answer provides the specific award(s) the book won, and the award is one of the major literary awards (Pulitzer Prize for Fiction, National Book Award for Fiction, or Booker Prize) from 2024 or 2025",
        parent=parent,
        critical=True,
    )

    selected = pick_award_for_checks(info.awards or [])
    has_award_with_sources = bool(selected and selected.name and selected.year and selected.urls)

    evaluator.add_custom_node(
        result=has_award_with_sources,
        id="Award_Information_exists",
        desc="Award information is provided with at least one supporting URL",
        parent=node,
        critical=True,
    )

    # Award is one of the allowed major awards
    allowed_leaf = evaluator.add_leaf(
        id="Award_Information_allowed_award",
        desc="The award is one of the allowed major awards (Pulitzer Prize for Fiction, National Book Award for Fiction, or Booker Prize)",
        parent=node,
        critical=True,
    )
    award_name = (selected.name if selected and selected.name else "")
    await evaluator.verify(
        claim=f"The award named '{award_name}' is one of the following: {', '.join(ALLOWED_AWARDS)}.",
        node=allowed_leaf,
        additional_instruction="Judge only the membership relationship; allow minor formatting variants. Do not require URL evidence for this logical membership check.",
    )

    # Award year is 2024 or 2025
    year_leaf = evaluator.add_leaf(
        id="Award_Information_year_2024_2025",
        desc="The award year is either 2024 or 2025",
        parent=node,
        critical=True,
    )
    award_year_str = (selected.year if selected and selected.year else "")
    await evaluator.verify(
        claim=f"The award year '{award_year_str}' is either 2024 or 2025.",
        node=year_leaf,
        additional_instruction="This is a simple logical check on the string; allow reasonable normalization such as extracting the 4-digit year.",
    )

    # Sources show the book actually WON that award (not just nominated/shortlisted)
    supported_leaf = evaluator.add_leaf(
        id="Award_Information_supported_by_sources",
        desc="Sources show that the book actually won the specified major award in 2024/2025 (not just nominated/shortlisted)",
        parent=node,
        critical=True,
    )
    title_display = first_non_empty_str(info.title, "the book")
    claim = (
        f"The cited source(s) explicitly indicate that '{title_display}' won the {award_name} in {award_year_str}, "
        f"as a winner (not merely nominated, longlisted, or shortlisted)."
    )
    award_sources = selected.urls if selected else []
    await evaluator.verify(
        claim=claim,
        node=supported_leaf,
        sources=award_sources,
        additional_instruction="Confirm explicit winner status for the book and the correct year on the provided webpage(s). If the page only indicates nomination/longlist/shortlist, this should be incorrect.",
    )


async def verify_publication_year_information(evaluator: Evaluator, parent, info: BookExtraction):
    node = evaluator.add_sequential(
        id="Publication_Year_Information",
        desc="The answer provides the specific year of publication, and it is between 2023 and 2024 (inclusive)",
        parent=parent,
        critical=True,
    )

    pub_year_present_with_source = bool(
        first_non_empty_str(info.publication_year) and (info.publication_year_urls and len(info.publication_year_urls) > 0)
    )
    evaluator.add_custom_node(
        result=pub_year_present_with_source,
        id="Publication_Year_Information_exists",
        desc="Publication year is provided with at least one supporting URL",
        parent=node,
        critical=True,
    )

    # In-range logical check
    pub_year_num = info.publication_year_number if info.publication_year_number is not None else safe_int(info.publication_year)
    in_range = bool(pub_year_num is not None and (PUB_YEAR_MIN <= pub_year_num <= PUB_YEAR_MAX))
    evaluator.add_custom_node(
        result=in_range,
        id="Publication_Year_Information_in_range",
        desc=f"Publication year ({pub_year_num if pub_year_num is not None else 'unknown'}) is between {PUB_YEAR_MIN} and {PUB_YEAR_MAX} inclusive",
        parent=node,
        critical=True,
    )

    # Sources support the publication year
    leaf = evaluator.add_leaf(
        id="Publication_Year_Information_supported_by_sources",
        desc="Sources support the claimed publication year",
        parent=node,
        critical=True,
    )
    title_display = first_non_empty_str(info.title, "the book")
    pub_year_str = first_non_empty_str(info.publication_year, str(pub_year_num) if pub_year_num else "")
    await evaluator.verify(
        claim=f"The cited source(s) indicate that the original publication year of '{title_display}' is {pub_year_str}.",
        node=leaf,
        sources=info.publication_year_urls,
        additional_instruction="Prefer the publisher's page or authoritative listings. If multiple editions exist, the year claimed in the answer must be supported.",
    )


async def verify_book_title_and_genre(evaluator: Evaluator, parent, info: BookExtraction):
    node = evaluator.add_sequential(
        id="Book_Title_and_Genre",
        desc="The answer provides the book's title and the book is categorized as a fiction novel",
        parent=parent,
        critical=True,
    )

    title_provided = bool(first_non_empty_str(info.title))
    evaluator.add_custom_node(
        result=title_provided,
        id="Book_Title_and_Genre_title_provided",
        desc="The book title is provided",
        parent=node,
        critical=True,
    )

    # We will verify "fiction novel" using any available credible URLs from the answer
    # Priority: publisher page > award page(s) > other metadata pages (publication/page count/language)
    award_urls_all = combine_sources(*[aw.urls for aw in (info.awards or [])])
    genre_sources = combine_sources(info.publisher_urls, award_urls_all, info.publication_year_urls, info.page_count_urls, info.language_urls)

    leaf = evaluator.add_leaf(
        id="Book_Title_and_Genre_fiction_supported",
        desc="Sources support that the book is a fiction novel",
        parent=node,
        critical=True,
    )
    title_display = first_non_empty_str(info.title, "the book")
    await evaluator.verify(
        claim=f"The cited source(s) categorize '{title_display}' as a fiction novel (or clearly as a novel in the fiction genre).",
        node=leaf,
        sources=genre_sources,
        additional_instruction="Treat 'novel' as implying fiction unless the page explicitly states non-fiction. Accept labels like 'literary fiction', 'novel', 'fiction'.",
    )


async def verify_author_education_information(evaluator: Evaluator, parent, info: BookExtraction):
    node = evaluator.add_sequential(
        id="Author_Education_Information",
        desc="The answer provides the author's name and highest degree with the institution name, and the degree is at least a master's degree from an accredited university",
        parent=parent,
        critical=True,
    )

    edu_present_with_source = bool(
        first_non_empty_str(info.author_name) and first_non_empty_str(info.author_degree) and first_non_empty_str(info.author_institution)
        and (info.author_education_urls and len(info.author_education_urls) > 0)
    )
    evaluator.add_custom_node(
        result=edu_present_with_source,
        id="Author_Education_Information_exists",
        desc="Author name, degree, institution are provided with at least one supporting URL",
        parent=node,
        critical=True,
    )

    # Degree is at least master's (logical check on the degree string)
    degree_ok = is_degree_string_at_least_masters(info.author_degree)
    evaluator.add_custom_node(
        result=degree_ok,
        id="Author_Education_Information_degree_at_least_masters",
        desc=f"Author's degree '{first_non_empty_str(info.author_degree)}' is at least a master's level",
        parent=node,
        critical=True,
    )

    # Sources support the degree + institution claim (and implicitly that the institution is accredited)
    leaf = evaluator.add_leaf(
        id="Author_Education_Information_supported_by_sources",
        desc="Sources support that the author holds the specified degree from the specified institution (accredited university)",
        parent=node,
        critical=True,
    )
    author = first_non_empty_str(info.author_name, "the author")
    deg = first_non_empty_str(info.author_degree, "a graduate degree")
    inst = first_non_empty_str(info.author_institution, "an accredited university")
    await evaluator.verify(
        claim=f"The cited source(s) indicate that {author} holds {deg} from {inst}.",
        node=leaf,
        sources=info.author_education_urls,
        additional_instruction="Prefer official bios, reputable profiles, or university sources. The institution should be a recognized/accredited university.",
    )


async def verify_page_count_information(evaluator: Evaluator, parent, info: BookExtraction):
    node = evaluator.add_sequential(
        id="Page_Count_Information",
        desc="The answer provides the specific number of pages, and it is between 100 and 350 pages",
        parent=parent,
        critical=True,
    )

    pages_present_with_source = bool(
        first_non_empty_str(info.page_count) and (info.page_count_urls and len(info.page_count_urls) > 0)
    )
    evaluator.add_custom_node(
        result=pages_present_with_source,
        id="Page_Count_Information_exists",
        desc="Page count is provided with at least one supporting URL",
        parent=node,
        critical=True,
    )

    pages_int = parse_page_int(info.page_count, info.page_count_number)
    in_pages_range = bool(pages_int is not None and (MIN_PAGES <= pages_int <= MAX_PAGES))
    evaluator.add_custom_node(
        result=in_pages_range,
        id="Page_Count_Information_in_range",
        desc=f"Page count ({pages_int if pages_int is not None else 'unknown'}) is between {MIN_PAGES} and {MAX_PAGES}",
        parent=node,
        critical=True,
    )

    # Sources support the page count
    leaf = evaluator.add_leaf(
        id="Page_Count_Information_supported_by_sources",
        desc="Sources support the claimed page count",
        parent=node,
        critical=True,
    )
    title_display = first_non_empty_str(info.title, "the book")
    pages_text = first_non_empty_str(info.page_count, str(pages_int) if pages_int is not None else "")
    await evaluator.verify(
        claim=f"The cited source(s) indicate that '{title_display}' has {pages_text} pages.",
        node=leaf,
        sources=info.page_count_urls,
        additional_instruction="Accept reasonable minor variations across editions if the answer's claim matches the source used in the answer.",
    )


async def verify_publisher_information(evaluator: Evaluator, parent, info: BookExtraction):
    node = evaluator.add_sequential(
        id="Publisher_Information",
        desc="The answer provides the publisher's name, and it is a major traditional publishing house",
        parent=parent,
        critical=True,
    )

    publisher_present_with_source = bool(
        first_non_empty_str(info.publisher) and (info.publisher_urls and len(info.publisher_urls) > 0)
    )
    evaluator.add_custom_node(
        result=publisher_present_with_source,
        id="Publisher_Information_exists",
        desc="Publisher is provided with at least one supporting URL showing it published the book",
        parent=node,
        critical=True,
    )

    # Verify that this publisher published the book (source = publisher_urls)
    leaf_pub_of_book = evaluator.add_leaf(
        id="Publisher_Information_published_this_book",
        desc="Sources show that the named publisher/imprint published the book",
        parent=node,
        critical=True,
    )
    title_display = first_non_empty_str(info.title, "the book")
    publisher_name = first_non_empty_str(info.publisher)
    await evaluator.verify(
        claim=f"The cited source(s) indicate that '{title_display}' was published by '{publisher_name}'.",
        node=leaf_pub_of_book,
        sources=info.publisher_urls,
        additional_instruction="If an imprint is named, treat it as the publisher of record for the book.",
    )

    # Verify "major traditional publishing house" (source = publisher_status_urls if present, else publisher_urls)
    status_sources = combine_sources(info.publisher_status_urls, info.publisher_urls)
    leaf_major = evaluator.add_leaf(
        id="Publisher_Information_major_traditional",
        desc="Sources support that the publisher is a major traditional publishing house",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The cited source(s) indicate that '{publisher_name}' is a major traditional publishing house "
            f"(e.g., part of or comparable to the Big Five, or an imprint under them)."
        ),
        node=leaf_major,
        sources=status_sources,
        additional_instruction="Wikipedia or reputable publishing industry sources are acceptable. Imprints under Big Five (e.g., PRH/Knopf, HarperCollins, Hachette, Macmillan, Simon & Schuster) count as major traditional publishing houses.",
    )


async def verify_language_availability(evaluator: Evaluator, parent, info: BookExtraction):
    node = evaluator.add_sequential(
        id="Language_Availability",
        desc="The book is available in English (either originally written in English or professionally translated into English)",
        parent=parent,
        critical=True,
    )

    language_claim_present = bool(first_non_empty_str(info.language))
    evaluator.add_custom_node(
        result=language_claim_present,
        id="Language_Availability_claim_present",
        desc="English availability is claimed in the answer",
        parent=node,
        critical=True,
    )

    # Try to verify via language_urls; if empty, fall back to publisher/title-related URLs
    award_urls_all = combine_sources(*[aw.urls for aw in (info.awards or [])])
    lang_sources = combine_sources(info.language_urls, info.publisher_urls, award_urls_all, info.publication_year_urls)

    leaf = evaluator.add_leaf(
        id="Language_Availability_english_supported",
        desc="Sources support that the book is available in English (original or professional translation)",
        parent=node,
        critical=True,
    )
    title_display = first_non_empty_str(info.title, "the book")
    await evaluator.verify(
        claim=f"The cited source(s) indicate that '{title_display}' is available in English (originally in English or translated into English).",
        node=leaf,
        sources=lang_sources,
        additional_instruction="Accept explicit 'Language: English', 'English edition', or clear mention of an English translation. Retailer or publisher pages are acceptable.",
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
    Evaluate an answer for the 'Award-Winning Contemporary Fiction Book' task using the Mind2Web2 framework.
    """
    # Initialize evaluator
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

    # Extract book info from the answer
    book_info: BookExtraction = await evaluator.extract(
        prompt=prompt_extract_book(),
        template_class=BookExtraction,
        extraction_name="book_extraction",
    )

    # Add high-level "book" critical node (per rubric)
    book_node = evaluator.add_parallel(
        id="Award-Winning_Contemporary_Fiction_Book",
        desc="The answer provides a complete identification of a book meeting all specified criteria, including all required information",
        parent=root,
        critical=True,
    )

    # Add reference/ground truth-like info for transparency
    evaluator.add_ground_truth({
        "allowed_awards": ALLOWED_AWARDS,
        "required_award_years": [2024, 2025],
        "publication_year_range": [PUB_YEAR_MIN, PUB_YEAR_MAX],
        "page_count_range": [MIN_PAGES, MAX_PAGES],
        "language_required": "English (original or professional translation)",
        "publisher_requirement": "Major traditional publishing house (Big Five or comparable; imprints acceptable).",
    }, gt_type="evaluation_requirements")

    # Build and run verification subtrees
    await verify_award_information(evaluator, book_node, book_info)
    await verify_publication_year_information(evaluator, book_node, book_info)
    await verify_book_title_and_genre(evaluator, book_node, book_info)
    await verify_author_education_information(evaluator, book_node, book_info)
    await verify_page_count_information(evaluator, book_node, book_info)
    await verify_publisher_information(evaluator, book_node, book_info)
    await verify_language_availability(evaluator, book_node, book_info)

    # Return structured evaluation summary
    return evaluator.get_summary()