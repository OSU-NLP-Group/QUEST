import asyncio
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "literary_awards_2024_2025"
TASK_DESCRIPTION = (
    "Identify the three distinct books that won the following major English-language literary awards in 2024 and 2025: "
    "the 2024 Booker Prize, the 2024 Pulitzer Prize for Fiction, the 2024 National Book Award for Fiction, and the 2025 Pulitzer Prize for Fiction. "
    "For each book, provide: (1) The complete title and author name, (2) The publishing imprint that published the book, "
    "(3) The parent publishing group of that imprint, (4) The ultimate parent company (one of the Big Five publishers), "
    "(5) All major literary awards won by the book (including award name and year), and (6) Reference URLs confirming the above information. "
    "Additionally, if available, include ISBN information for at least one format, publication date, award eligibility confirmation, "
    "and any notable characteristics of the book. Note that one book won multiple awards, so there are only three distinct books across the four award categories mentioned."
)

EXPECTED_BIG_FIVE = [
    "Penguin Random House",
    "HarperCollins",
    "Simon & Schuster",
    "Hachette Book Group",
    "Macmillan",
]

# Ground truth mapping (used to guide verification claims)
EXPECTED_WINNERS = {
    "booker_2024": {"title": "Orbital", "author": "Samantha Harvey"},
    "pulitzer_2024": {"title": "Night Watch", "author": "Jayne Anne Phillips"},
    "nba_2024_fiction": {"title": "James", "author": "Percival Everett"},
    "pulitzer_2025": {"title": "James", "author": "Percival Everett"},
}

NBA_2024_START = date(2023, 12, 1)
NBA_2024_END = date(2024, 11, 30)

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class AwardEntry(BaseModel):
    name: Optional[str] = None
    year: Optional[str] = None
    notes: Optional[str] = None


class ISBNEntry(BaseModel):
    isbn: Optional[str] = None
    format: Optional[str] = None  # hardcover/paperback/ebook/audio etc.


class BookExtract(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None

    imprint: Optional[str] = None
    parent_publishing_group: Optional[str] = None
    ultimate_parent_company: Optional[str] = None

    awards: List[AwardEntry] = Field(default_factory=list)

    refs_title_author: List[str] = Field(default_factory=list)
    refs_publisher_chain: List[str] = Field(default_factory=list)
    refs_awards: List[str] = Field(default_factory=list)

    publication_date: Optional[str] = None
    refs_publication: List[str] = Field(default_factory=list)

    isbns: List[ISBNEntry] = Field(default_factory=list)
    refs_isbn: List[str] = Field(default_factory=list)

    notable_characteristics: List[str] = Field(default_factory=list)
    refs_characteristics: List[str] = Field(default_factory=list)


class AnswerExtraction(BaseModel):
    books: List[BookExtract] = Field(default_factory=list)
    big_five_list_in_answer: List[str] = Field(default_factory=list)
    big_five_refs: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    From the provided answer, extract structured information for up to 5 mentioned books that relate to the specified award winners across 2024 and 2025.
    Focus on the winners: 2024 Booker Prize (book + author), 2024 Pulitzer Prize for Fiction (book + author),
    2024 National Book Award for Fiction (book + author), and 2025 Pulitzer Prize for Fiction (book + author).
    Note that there should only be three distinct books because one book won both the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction.

    For each book found in the answer, extract the following fields (JSON keys must match exactly):
    - title: Complete title string exactly as in the answer
    - author: Complete author name exactly as in the answer
    - imprint: Publishing imprint that published this book (as stated in the answer)
    - parent_publishing_group: Parent publishing group of that imprint (as stated in the answer)
    - ultimate_parent_company: Ultimate parent company; this should be one of the Big Five (as stated in the answer)
    - awards: an array of { "name": string, "year": string, "notes": string or null } representing awards won by the book (name + year must be included if the answer claims them)
    - refs_title_author: array of authoritative URLs cited for confirming title and author (publisher page, award page, etc.)
    - refs_publisher_chain: array of authoritative URLs cited for confirming the imprint -> group -> ultimate parent company chain
    - refs_awards: array of authoritative URLs cited for confirming the awards won
    - publication_date: publication date or at least publication year as a free-form string (if provided)
    - refs_publication: array of URLs cited for confirming the publication date (or year)
    - isbns: an array of { "isbn": string, "format": string } for any provided ISBNs (at least one format if available)
    - refs_isbn: array of URLs cited for confirming any provided ISBNs
    - notable_characteristics: array of notable facts/characteristics about the book (e.g., first space-set winner, page count, shortest/second shortest winner, etc.) if claimed
    - refs_characteristics: array of URLs cited for confirming notable characteristics

    Additionally, if the answer explicitly lists the Big Five publishers, extract:
    - big_five_list_in_answer: array of publisher names exactly as listed in the answer
    - big_five_refs: array of URLs cited for confirming Big Five definitions or membership (if any are provided)

    Important:
    - Only extract information that is explicitly present in the answer text.
    - If a field is missing, set it to null or an empty list as appropriate.
    - Do not invent URLs; include only URLs explicitly provided in the answer (plain URL or markdown link).
    - Preserve the exact spelling/casing of titles and names from the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_title(s: Optional[str]) -> str:
    if not s:
        return ""
    return "".join(ch for ch in s.lower().strip() if ch.isalnum() or ch.isspace())


def normalize_author(s: Optional[str]) -> str:
    if not s:
        return ""
    return "".join(ch for ch in s.lower().strip() if ch.isalnum() or ch.isspace())


def titles_distinct(books: List[BookExtract]) -> int:
    seen = set()
    for b in books:
        t = normalize_title(b.title)
        if t:
            seen.add(t)
    return len(seen)


def select_book(books: List[BookExtract], expected_title: str, expected_author: str) -> BookExtract:
    target_t = normalize_title(expected_title)
    target_a = normalize_author(expected_author)

    # Try title match first
    for b in books:
        if normalize_title(b.title) == target_t:
            return b
    # Fallback to author match
    for b in books:
        if normalize_author(b.author) == target_a:
            return b
    # Fallback: return empty placeholder
    return BookExtract()


def normalize_award_name(name: Optional[str]) -> str:
    if not name:
        return ""
    s = name.lower()
    s = s.replace("&", "and")
    return "".join(ch for ch in s if ch.isalnum() or ch.isspace())


def award_covered(books: List[BookExtract], key: str, year: str) -> bool:
    """
    Check if within extracted books there exists an award entry that matches the expected key and year.
    key: one of ["booker", "pulitzer fiction", "national book award fiction"]
    """
    key_norm = normalize_award_name(key)
    year_norm = "".join(ch for ch in year if ch.isdigit())
    for b in books:
        for a in b.awards:
            n = normalize_award_name(a.name)
            y = "".join(ch for ch in (a.year or "") if ch.isdigit())
            if key_norm in n and year_norm == y:
                return True
    return False


MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}


def parse_publication_date(d: Optional[str]) -> Optional[date]:
    if not d:
        return None
    s = d.strip()
    # ISO-like YYYY-MM-DD
    try:
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            y = int(s[0:4])
            m = int(s[5:7])
            dd = int(s[8:10])
            return date(y, m, dd)
    except Exception:
        pass
    # "Month DD, YYYY" or "Month YYYY"
    try:
        parts = s.replace(",", " ").split()
        parts = [p for p in parts if p]
        # Month Day Year or Month Year
        if len(parts) >= 2:
            month = MONTHS.get(parts[0].lower())
            if month:
                if len(parts) == 3:
                    day = int(parts[1])
                    year = int(parts[2])
                    return date(year, month, day)
                elif len(parts) == 2:
                    year = int(parts[1])
                    # Assume mid-month if only year/month
                    return date(year, month, 15)
    except Exception:
        pass
    # Year only
    try:
        if len(s) == 4 and s.isdigit():
            return date(int(s), 6, 15)  # mid-year placeholder
    except Exception:
        pass
    return None


def is_isbn13(s: Optional[str]) -> bool:
    if not s:
        return False
    digits = "".join(ch for ch in s if ch.isdigit())
    return len(digits) == 13


def is_isbn13_valid_for_list(isbns: List[ISBNEntry]) -> bool:
    """
    Validity per constraint:
    - If ISBN-13 is provided, it has 13 digits (ignore hyphens/spaces).
    - If multiple formats are provided, ISBNs are not reused across formats.
    """
    if not isbns:
        return False
    # Check lengths for 13-digit entries
    ok_len = True
    for entry in isbns:
        if entry.isbn and ("-" in entry.isbn or len("".join(ch for ch in entry.isbn if ch.isdigit())) in (10, 13)):
            # If it's 13-digit style, ensure it's exactly 13 digits
            digits = "".join(ch for ch in entry.isbn if ch.isdigit())
            if len(digits) == 13:
                continue
            # If provided entry looks like ISBN-10 (10 digits), this check focuses on 13-digit when provided;
            # we'll allow non-13 digits entries, but ensure any 13-digit ones are correctly 13 digits.
            if len(digits) != 10:
                ok_len = False
        elif entry.isbn:
            digits = "".join(ch for ch in entry.isbn if ch.isdigit())
            if len(digits) not in (10, 13):
                ok_len = False

    # Check reuse across formats
    format_map: Dict[str, List[str]] = {}
    for entry in isbns:
        if not entry.format or not entry.isbn:
            continue
        digits = "".join(ch for ch in entry.isbn if ch.isdigit())
        format_map.setdefault(entry.format.lower().strip(), []).append(digits)

    # If multiple formats exist, ensure digits not reused across formats
    all_formats = list(format_map.keys())
    if len(all_formats) > 1:
        # Build reverse map
        seen_digits: Dict[str, str] = {}
        for fmt, digits_list in format_map.items():
            for dg in digits_list:
                if dg in seen_digits and seen_digits[dg] != fmt:
                    return False
                seen_digits[dg] = fmt

    return ok_len


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_global_requirements(
    evaluator: Evaluator,
    parent: VerificationNode,
    extraction: AnswerExtraction,
    agent_answer: str
) -> VerificationNode:
    global_node = evaluator.add_parallel(
        id="Global_Requirements",
        desc="Cross-cutting requirements that apply to the whole response",
        parent=parent,
        critical=True
    )

    # Distinct book count must be exactly three
    exact_three = titles_distinct(extraction.books) == 3
    evaluator.add_custom_node(
        result=exact_three,
        id="Distinct_Book_Count",
        desc="Response identifies exactly three distinct books while covering all four specified award categories (one book wins multiple awards).",
        parent=global_node,
        critical=True
    )

    # Coverage of all award categories (programmatic check)
    cov_booker_2024 = award_covered(extraction.books, "booker prize", "2024")
    cov_pulitzer_2024 = award_covered(extraction.books, "pulitzer fiction", "2024")
    cov_nba_2024 = award_covered(extraction.books, "national book award fiction", "2024")
    cov_pulitzer_2025 = award_covered(extraction.books, "pulitzer fiction", "2025")
    coverage_all = cov_booker_2024 and cov_pulitzer_2024 and cov_nba_2024 and cov_pulitzer_2025

    evaluator.add_custom_node(
        result=coverage_all,
        id="Coverage_Of_All_Award_Categories",
        desc="Response explicitly covers: 2024 Booker Prize, 2024 Pulitzer Prize for Fiction, 2024 National Book Award for Fiction, and 2025 Pulitzer Prize for Fiction winners.",
        parent=global_node,
        critical=True
    )

    # Big Five list must be listed exactly as specified (use simple verification on the answer text)
    bigfive_leaf = evaluator.add_leaf(
        id="Big_Five_List",
        desc="Response lists the Big Five publishers exactly as: Penguin Random House, HarperCollins, Simon & Schuster, Hachette Book Group, and Macmillan.",
        parent=global_node,
        critical=True
    )
    claim_big_five = (
        "The answer lists the Big Five publishers exactly as: "
        "Penguin Random House, HarperCollins, Simon & Schuster, Hachette Book Group, and Macmillan."
    )
    await evaluator.verify(
        claim=claim_big_five,
        node=bigfive_leaf,
        additional_instruction=(
            "Check the answer text itself to confirm an explicit list matching the required five names exactly. "
            "Minor punctuation differences are okay, but the five names must appear as specified."
        )
    )

    return global_node


async def verify_book_1_james(
    evaluator: Evaluator,
    parent: VerificationNode,
    extraction: AnswerExtraction
) -> VerificationNode:
    expected = EXPECTED_WINNERS["nba_2024_fiction"]
    expected2 = EXPECTED_WINNERS["pulitzer_2025"]
    # Use the same book object for both awards (James)
    book = select_book(extraction.books, expected["title"], expected["author"])

    book_node = evaluator.add_parallel(
        id="Book_1_James",
        desc="Book that won both the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction",
        parent=parent,
        critical=False
    )

    # Title/Author group
    ta_node = evaluator.add_parallel(
        id="Book_1_Title_Author",
        desc="Correct title/author with authoritative reference URL(s)",
        parent=book_node,
        critical=True
    )
    # References existence (critical gate)
    ta_refs_ok = bool(book.refs_title_author)
    evaluator.add_custom_node(
        result=ta_refs_ok,
        id="Book_1_TitleAuthor_Refs",
        desc="Provide authoritative reference URL(s) confirming title and author.",
        parent=ta_node,
        critical=True
    )
    # Title check
    title_leaf = evaluator.add_leaf(
        id="Book_1_Title",
        desc="Title is 'James'.",
        parent=ta_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page shows the book titled 'James'.",
        node=title_leaf,
        sources=book.refs_title_author,
        additional_instruction="Confirm the title 'James' appears clearly on the referenced page(s)."
    )
    # Author check
    author_leaf = evaluator.add_leaf(
        id="Book_1_Author",
        desc="Author is Percival Everett.",
        parent=ta_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page shows that the author of 'James' is Percival Everett.",
        node=author_leaf,
        sources=book.refs_title_author,
        additional_instruction="Confirm the author attribution Percival Everett for the book 'James'."
    )

    # Publisher chain
    pc_node = evaluator.add_parallel(
        id="Book_1_Publisher_Chain",
        desc="Imprint, parent publishing group, and ultimate parent company with references, per constraints",
        parent=book_node,
        critical=True
    )
    # References gate
    pc_refs_ok = bool(book.refs_publisher_chain)
    evaluator.add_custom_node(
        result=pc_refs_ok,
        id="Book_1_PublisherChain_Refs",
        desc="Provide authoritative reference URL(s) confirming the imprint/group/ultimate-parent claims.",
        parent=pc_node,
        critical=True
    )
    # Imprint: Doubleday
    imprint_leaf = evaluator.add_leaf(
        id="Book_1_Imprint",
        desc="Publishing imprint is Doubleday.",
        parent=pc_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page indicates the publishing imprint for 'James' is Doubleday.",
        node=imprint_leaf,
        sources=book.refs_publisher_chain,
        additional_instruction="Confirm the imprint 'Doubleday' on publisher pages or authoritative sources."
    )
    # Imprint to group
    itg_leaf = evaluator.add_leaf(
        id="Book_1_Imprint_To_Group",
        desc="Doubleday is identified as an imprint of the Knopf Doubleday Publishing Group.",
        parent=pc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Doubleday is an imprint under the Knopf Doubleday Publishing Group.",
        node=itg_leaf,
        sources=book.refs_publisher_chain,
        additional_instruction="Verify imprint-to-group relationship via publisher pages or reliable industry sources."
    )
    # Group to PRH
    gtp_leaf = evaluator.add_leaf(
        id="Book_1_Group_To_PRH",
        desc="Knopf Doubleday Publishing Group is identified as part of Penguin Random House.",
        parent=pc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Knopf Doubleday Publishing Group is part of Penguin Random House.",
        node=gtp_leaf,
        sources=book.refs_publisher_chain,
        additional_instruction="Confirm the group is a division of Penguin Random House."
    )
    # PRH is Big Five
    prh_bf_leaf = evaluator.add_leaf(
        id="Book_1_PRH_BigFive",
        desc="Penguin Random House is identified as one of the Big Five publishers.",
        parent=pc_node,
        critical=True
    )
    sources_bf = extraction.big_five_refs if extraction.big_five_refs else book.refs_publisher_chain
    await evaluator.verify(
        claim="Penguin Random House is one of the Big Five publishers.",
        node=prh_bf_leaf,
        sources=sources_bf,
        additional_instruction="Use industry references or authoritative articles to confirm PRH's Big Five status."
    )

    # Awards
    aw_node = evaluator.add_parallel(
        id="Book_1_Awards",
        desc="Awards won (with award name + year) and references, including the required multi-award condition",
        parent=book_node,
        critical=True
    )
    # References gate
    aw_refs_ok = bool(book.refs_awards)
    evaluator.add_custom_node(
        result=aw_refs_ok,
        id="Book_1_Award_Refs",
        desc="Provide authoritative reference URL(s) confirming the awards claimed.",
        parent=aw_node,
        critical=True
    )
    # NBA 2024
    nba_leaf = evaluator.add_leaf(
        id="Book_1_NBA_2024",
        desc="Confirms the book won the 2024 National Book Award for Fiction (award name + year).",
        parent=aw_node,
        critical=True
    )
    await evaluator.verify(
        claim="The book 'James' won the 2024 National Book Award for Fiction.",
        node=nba_leaf,
        sources=book.refs_awards,
        additional_instruction="Confirm via the National Book Foundation page or equivalent authoritative source."
    )
    # Pulitzer 2025
    pul2025_leaf = evaluator.add_leaf(
        id="Book_1_Pulitzer_2025",
        desc="Confirms the book won the 2025 Pulitzer Prize for Fiction (award name + year).",
        parent=aw_node,
        critical=True
    )
    await evaluator.verify(
        claim="The book 'James' won the 2025 Pulitzer Prize for Fiction.",
        node=pul2025_leaf,
        sources=book.refs_awards,
        additional_instruction="Confirm via the Pulitzer.org page or equivalent authoritative source."
    )
    # Multi-award recognition (answer explicitly recognizes same book)
    multi_leaf = evaluator.add_leaf(
        id="Book_1_MultiAward_Recognition",
        desc="Explicitly recognizes that 'James' is the same book winning both the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction.",
        parent=aw_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly recognizes that 'James' won both the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction.",
        node=multi_leaf,
        additional_instruction="Check the answer text itself; do not rely on external pages."
    )
    # Other major awards listed (non-critical): pass if none claimed, or ensure name+year provided
    others_ok = True
    other_awards = [a for a in book.awards if a and a.name and a.year]
    # If any awards are claimed beyond the required ones, ensure they have both name and year
    provided_awards_total = len(book.awards)
    required_pair = {("National Book Award for Fiction", "2024"), ("Pulitzer Prize for Fiction", "2025")}
    # If provided_awards_total > 2, ensure all extra entries have name+year non-empty
    if provided_awards_total > 2:
        # validate every award entry has name and year strings with digits for year
        for a in book.awards:
            if not a.name or not a.year or not any(ch.isdigit() for ch in a.year):
                others_ok = False
                break
    evaluator.add_custom_node(
        result=others_ok,
        id="Book_1_Other_Major_Awards_Listed",
        desc="Lists any other major literary awards won by the book (if any are claimed, include award name + year).",
        parent=aw_node,
        critical=False
    )

    # Publication & eligibility (non-critical)
    pub_node = evaluator.add_parallel(
        id="Book_1_Publication_And_Eligibility",
        desc="Publication date and National Book Award eligibility check per constraint (if provided), with references",
        parent=book_node,
        critical=False
    )
    pub_leaf = evaluator.add_leaf(
        id="Book_1_Publication_Date",
        desc="Provide publication date (or at least publication year) with a reference URL.",
        parent=pub_node,
        critical=False
    )
    pub_claim = f"The publication date (or year) for 'James' is '{book.publication_date}'." if book.publication_date else "The publication date (or year) for 'James' is provided."
    await evaluator.verify(
        claim=pub_claim,
        node=pub_leaf,
        sources=book.refs_publication,
        additional_instruction="Confirm the publication date/year for 'James' using the provided reference URLs."
    )
    # Eligibility window (non-critical) - programmatic check if date is parseable
    pub_dt = parse_publication_date(book.publication_date)
    in_window = False
    if pub_dt:
        in_window = (NBA_2024_START <= pub_dt <= NBA_2024_END) or (pub_dt.year == 2024)
    evaluator.add_custom_node(
        result=in_window,
        id="Book_1_NBA_Eligibility_Window_Check",
        desc="If publication date is provided, confirm it falls within the National Book Award eligibility window for the 2024 awards (Dec 1, 2023–Nov 30, 2024).",
        parent=pub_node,
        critical=False
    )

    # ISBN (non-critical)
    isbn_node = evaluator.add_parallel(
        id="Book_1_ISBN",
        desc="ISBN information (if available) for at least one format, with basic validity checks and references",
        parent=book_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=bool(book.isbns),
        id="Book_1_ISBN_Provided",
        desc="Provide ISBN for at least one format (hardcover/paperback/ebook) if available, and specify the format.",
        parent=isbn_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=is_isbn13_valid_for_list(book.isbns) if book.isbns else False,
        id="Book_1_ISBN_Format_Validity",
        desc="If ISBN-13 is provided, it has 13 digits; if multiple formats are provided, ISBNs are not reused across formats.",
        parent=isbn_node,
        critical=False
    )
    isbn_refs_leaf = evaluator.add_leaf(
        id="Book_1_ISBN_Refs",
        desc="Provide reference URL(s) confirming ISBN information (if ISBN is provided).",
        parent=isbn_node,
        critical=False
    )
    await evaluator.verify(
        claim="The provided reference URL(s) confirm the ISBN information for 'James'.",
        node=isbn_refs_leaf,
        sources=book.refs_isbn,
        additional_instruction="Verify ISBN(s) for 'James' via publisher pages or authoritative catalogs."
    )

    return book_node


async def verify_book_2_night_watch(
    evaluator: Evaluator,
    parent: VerificationNode,
    extraction: AnswerExtraction
) -> VerificationNode:
    expected = EXPECTED_WINNERS["pulitzer_2024"]
    book = select_book(extraction.books, expected["title"], expected["author"])

    book_node = evaluator.add_parallel(
        id="Book_2_Night_Watch",
        desc="Book that won the 2024 Pulitzer Prize for Fiction",
        parent=parent,
        critical=False
    )

    # Title/Author
    ta_node = evaluator.add_parallel(
        id="Book_2_Title_Author",
        desc="Correct title/author with authoritative reference URL(s)",
        parent=book_node,
        critical=True
    )
    ta_refs_ok = bool(book.refs_title_author)
    evaluator.add_custom_node(
        result=ta_refs_ok,
        id="Book_2_TitleAuthor_Refs",
        desc="Provide authoritative reference URL(s) confirming title and author.",
        parent=ta_node,
        critical=True
    )
    title_leaf = evaluator.add_leaf(
        id="Book_2_Title",
        desc="Title is 'Night Watch'.",
        parent=ta_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page shows the book titled 'Night Watch'.",
        node=title_leaf,
        sources=book.refs_title_author,
        additional_instruction="Confirm the title 'Night Watch' appears on the referenced page(s)."
    )
    author_leaf = evaluator.add_leaf(
        id="Book_2_Author",
        desc="Author is Jayne Anne Phillips.",
        parent=ta_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page shows that the author of 'Night Watch' is Jayne Anne Phillips.",
        node=author_leaf,
        sources=book.refs_title_author,
        additional_instruction="Confirm the author attribution Jayne Anne Phillips."
    )

    # Publisher chain
    pc_node = evaluator.add_parallel(
        id="Book_2_Publisher_Chain",
        desc="Imprint, parent publishing group, and ultimate parent company with references, per constraints",
        parent=book_node,
        critical=True
    )
    pc_refs_ok = bool(book.refs_publisher_chain)
    evaluator.add_custom_node(
        result=pc_refs_ok,
        id="Book_2_PublisherChain_Refs",
        desc="Provide authoritative reference URL(s) confirming the imprint/group/ultimate-parent claims.",
        parent=pc_node,
        critical=True
    )
    imprint_leaf = evaluator.add_leaf(
        id="Book_2_Imprint",
        desc="Publishing imprint is Knopf.",
        parent=pc_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page indicates the publishing imprint for 'Night Watch' is Knopf.",
        node=imprint_leaf,
        sources=book.refs_publisher_chain,
        additional_instruction="Confirm 'Alfred A. Knopf' (Knopf) imprint attribution."
    )
    itg_leaf = evaluator.add_leaf(
        id="Book_2_Imprint_To_Group",
        desc="Knopf is identified as an imprint of the Knopf Doubleday Publishing Group.",
        parent=pc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Knopf is an imprint under the Knopf Doubleday Publishing Group.",
        node=itg_leaf,
        sources=book.refs_publisher_chain,
        additional_instruction="Verify imprint-to-group relationship."
    )
    gtp_leaf = evaluator.add_leaf(
        id="Book_2_Group_To_PRH",
        desc="Knopf Doubleday Publishing Group is identified as part of Penguin Random House.",
        parent=pc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Knopf Doubleday Publishing Group is part of Penguin Random House.",
        node=gtp_leaf,
        sources=book.refs_publisher_chain,
        additional_instruction="Confirm group-to-ultimate parent relationship."
    )
    prh_bf_leaf = evaluator.add_leaf(
        id="Book_2_PRH_BigFive",
        desc="Penguin Random House is identified as one of the Big Five publishers.",
        parent=pc_node,
        critical=True
    )
    sources_bf = extraction.big_five_refs if extraction.big_five_refs else book.refs_publisher_chain
    await evaluator.verify(
        claim="Penguin Random House is one of the Big Five publishers.",
        node=prh_bf_leaf,
        sources=sources_bf,
        additional_instruction="Verify PRH Big Five status via authoritative sources."
    )

    # Awards
    aw_node = evaluator.add_parallel(
        id="Book_2_Awards",
        desc="Awards won (with award name + year) and references",
        parent=book_node,
        critical=True
    )
    aw_refs_ok = bool(book.refs_awards)
    evaluator.add_custom_node(
        result=aw_refs_ok,
        id="Book_2_Award_Refs",
        desc="Provide authoritative reference URL(s) confirming the awards claimed.",
        parent=aw_node,
        critical=True
    )
    pul_leaf = evaluator.add_leaf(
        id="Book_2_Pulitzer_2024",
        desc="Confirms the book won the 2024 Pulitzer Prize for Fiction (award name + year).",
        parent=aw_node,
        critical=True
    )
    await evaluator.verify(
        claim="The book 'Night Watch' won the 2024 Pulitzer Prize for Fiction.",
        node=pul_leaf,
        sources=book.refs_awards,
        additional_instruction="Confirm via Pulitzer.org or equivalent authoritative source."
    )
    # Other major awards listed (non-critical)
    others_ok = True
    if len(book.awards) > 1:
        for a in book.awards:
            if not a.name or not a.year or not any(ch.isdigit() for ch in a.year):
                others_ok = False
                break
    evaluator.add_custom_node(
        result=others_ok,
        id="Book_2_Other_Major_Awards_Listed",
        desc="Lists any other major literary awards won by the book (if any are claimed, include award name + year).",
        parent=aw_node,
        critical=False
    )

    # Publication details (non-critical)
    pub_node = evaluator.add_parallel(
        id="Book_2_Publication_Details",
        desc="Publication date (optional) with references",
        parent=book_node,
        critical=False
    )
    pub_leaf = evaluator.add_leaf(
        id="Book_2_Publication_Date",
        desc="Provide publication date (or at least publication year) with a reference URL.",
        parent=pub_node,
        critical=False
    )
    pub_claim = f"The publication date (or year) for 'Night Watch' is '{book.publication_date}'." if book.publication_date else "The publication date (or year) for 'Night Watch' is provided."
    await evaluator.verify(
        claim=pub_claim,
        node=pub_leaf,
        sources=book.refs_publication,
        additional_instruction="Confirm the publication date/year for 'Night Watch' using the provided URLs."
    )

    # ISBN (non-critical)
    isbn_node = evaluator.add_parallel(
        id="Book_2_ISBN",
        desc="ISBN information (if available) for at least one format, with basic validity checks and references",
        parent=book_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=bool(book.isbns),
        id="Book_2_ISBN_Provided",
        desc="Provide ISBN for at least one format (hardcover/paperback/ebook) if available, and specify the format.",
        parent=isbn_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=is_isbn13_valid_for_list(book.isbns) if book.isbns else False,
        id="Book_2_ISBN_Format_Validity",
        desc="If ISBN-13 is provided, it has 13 digits; if multiple formats are provided, ISBNs are not reused across formats.",
        parent=isbn_node,
        critical=False
    )
    isbn_refs_leaf = evaluator.add_leaf(
        id="Book_2_ISBN_Refs",
        desc="Provide reference URL(s) confirming ISBN information (if ISBN is provided).",
        parent=isbn_node,
        critical=False
    )
    await evaluator.verify(
        claim="The provided reference URL(s) confirm the ISBN information for 'Night Watch'.",
        node=isbn_refs_leaf,
        sources=book.refs_isbn,
        additional_instruction="Verify ISBN(s) via publisher pages or authoritative catalogs."
    )

    return book_node


async def verify_book_3_orbital(
    evaluator: Evaluator,
    parent: VerificationNode,
    extraction: AnswerExtraction
) -> VerificationNode:
    expected = EXPECTED_WINNERS["booker_2024"]
    book = select_book(extraction.books, expected["title"], expected["author"])

    book_node = evaluator.add_parallel(
        id="Book_3_Orbital",
        desc="Book that won the 2024 Booker Prize",
        parent=parent,
        critical=False
    )

    # Title/Author
    ta_node = evaluator.add_parallel(
        id="Book_3_Title_Author",
        desc="Correct title/author with authoritative reference URL(s)",
        parent=book_node,
        critical=True
    )
    ta_refs_ok = bool(book.refs_title_author)
    evaluator.add_custom_node(
        result=ta_refs_ok,
        id="Book_3_TitleAuthor_Refs",
        desc="Provide authoritative reference URL(s) confirming title and author.",
        parent=ta_node,
        critical=True
    )
    title_leaf = evaluator.add_leaf(
        id="Book_3_Title",
        desc="Title is 'Orbital'.",
        parent=ta_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page shows the book titled 'Orbital'.",
        node=title_leaf,
        sources=book.refs_title_author,
        additional_instruction="Confirm the title 'Orbital' appears on the referenced page(s)."
    )
    author_leaf = evaluator.add_leaf(
        id="Book_3_Author",
        desc="Author is Samantha Harvey.",
        parent=ta_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page shows that the author of 'Orbital' is Samantha Harvey.",
        node=author_leaf,
        sources=book.refs_title_author,
        additional_instruction="Confirm the author attribution Samantha Harvey."
    )

    # Publisher chain
    pc_node = evaluator.add_parallel(
        id="Book_3_Publisher_Chain",
        desc="Imprint, parent publishing group, and ultimate parent company (Big Five) with references",
        parent=book_node,
        critical=True
    )
    pc_refs_ok = bool(book.refs_publisher_chain)
    evaluator.add_custom_node(
        result=pc_refs_ok,
        id="Book_3_PublisherChain_Refs",
        desc="Provide authoritative reference URL(s) confirming the imprint/group/ultimate-parent claims.",
        parent=pc_node,
        critical=True
    )
    imprint_leaf = evaluator.add_leaf(
        id="Book_3_Imprint",
        desc="Publishing imprint is Jonathan Cape (UK).",
        parent=pc_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page indicates the publishing imprint for 'Orbital' is Jonathan Cape.",
        node=imprint_leaf,
        sources=book.refs_publisher_chain,
        additional_instruction="Confirm Jonathan Cape (UK) imprint attribution."
    )
    parent_group_leaf = evaluator.add_leaf(
        id="Book_3_Parent_Publishing_Group",
        desc="Provide the parent publishing group of the imprint (name) with a reference URL (no hard-coded expected name).",
        parent=pc_node,
        critical=True
    )
    parent_claim = "This page identifies the parent publishing group for the Jonathan Cape imprint."
    await evaluator.verify(
        claim=parent_claim,
        node=parent_group_leaf,
        sources=book.refs_publisher_chain,
        additional_instruction="Confirm the named parent publishing group for Jonathan Cape using authoritative sources."
    )
    ultimate_leaf = evaluator.add_leaf(
        id="Book_3_Ultimate_Parent_Company",
        desc="Provide the ultimate parent company and confirm it is one of the Big Five (using the Big Five list constraint) with reference URL(s).",
        parent=pc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The ultimate parent company for Jonathan Cape is Penguin Random House, which is one of the Big Five publishers.",
        node=ultimate_leaf,
        sources=extraction.big_five_refs if extraction.big_five_refs else book.refs_publisher_chain,
        additional_instruction="Confirm PRH ultimate parent and Big Five status via authoritative sources."
    )

    # Awards
    aw_node = evaluator.add_parallel(
        id="Book_3_Awards",
        desc="Awards won (with award name + year) and required Booker announcement details, with references",
        parent=book_node,
        critical=True
    )
    aw_refs_ok = bool(book.refs_awards)
    evaluator.add_custom_node(
        result=aw_refs_ok,
        id="Book_3_Award_Refs",
        desc="Provide authoritative reference URL(s) confirming the award win and announcement details.",
        parent=aw_node,
        critical=True
    )
    booker_leaf = evaluator.add_leaf(
        id="Book_3_Booker_2024",
        desc="Confirms the book won the 2024 Booker Prize (award name + year).",
        parent=aw_node,
        critical=True
    )
    await evaluator.verify(
        claim="The book 'Orbital' won the 2024 Booker Prize.",
        node=booker_leaf,
        sources=book.refs_awards,
        additional_instruction="Confirm via the Booker Prize official site or other authoritative source."
    )
    announce_leaf = evaluator.add_leaf(
        id="Book_3_Booker_Announcement_Details",
        desc="States the Booker Prize 2024 announcement was on Nov 12, 2024, at Old Billingsgate in London.",
        parent=aw_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Booker Prize 2024 winner announcement took place on November 12, 2024, at Old Billingsgate in London.",
        node=announce_leaf,
        sources=book.refs_awards,
        additional_instruction="Confirm these announcement details on the Booker Prize site or equivalent authoritative coverage."
    )

    # Notable characteristics
    nc_node = evaluator.add_parallel(
        id="Book_3_Notable_Characteristics",
        desc="Notable characteristics required by constraints (with references)",
        parent=book_node,
        critical=True
    )
    nc_refs_ok = bool(book.refs_characteristics)
    evaluator.add_custom_node(
        result=nc_refs_ok,
        id="Book_3_Characteristics_Refs",
        desc="Provide reference URL(s) confirming the notable characteristics.",
        parent=nc_node,
        critical=True
    )
    first_space_leaf = evaluator.add_leaf(
        id="Book_3_First_Space_Set_Winner",
        desc="Notes that 'Orbital' is the first book set in space to win the Booker Prize.",
        parent=nc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Orbital is the first book set in space to win the Booker Prize.",
        node=first_space_leaf,
        sources=book.refs_characteristics,
        additional_instruction="Confirm via official Booker site or authoritative articles."
    )
    pages_leaf = evaluator.add_leaf(
        id="Book_3_Page_Count_Shortness",
        desc="Notes the book is approximately 136 pages and is the second shortest Booker Prize winner.",
        parent=nc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Orbital is approximately 136 pages and is the second shortest Booker Prize winner.",
        node=pages_leaf,
        sources=book.refs_characteristics,
        additional_instruction="Confirm page count and shortest/second shortest status via authoritative sources."
    )

    # Publication & ISBN (non-critical)
    pub_node = evaluator.add_parallel(
        id="Book_3_Publication_And_ISBN",
        desc="Publication date and ISBN details if available (optional) with references",
        parent=book_node,
        critical=False
    )
    pub_leaf = evaluator.add_leaf(
        id="Book_3_Publication_Date",
        desc="Provide publication date (or at least publication year) with a reference URL.",
        parent=pub_node,
        critical=False
    )
    pub_claim = f"The publication date (or year) for 'Orbital' is '{book.publication_date}'." if book.publication_date else "The publication date (or year) for 'Orbital' is provided."
    await evaluator.verify(
        claim=pub_claim,
        node=pub_leaf,
        sources=book.refs_publication,
        additional_instruction="Confirm the publication date/year for 'Orbital' using the provided URLs."
    )
    isbn_provided_leaf = evaluator.add_custom_node(
        result=bool(book.isbns),
        id="Book_3_ISBN_Provided",
        desc="Provide ISBN for at least one format (hardcover/paperback/ebook) if available, and specify the format.",
        parent=pub_node,
        critical=False
    )
    isbn_valid_leaf = evaluator.add_custom_node(
        result=is_isbn13_valid_for_list(book.isbns) if book.isbns else False,
        id="Book_3_ISBN_Format_Validity",
        desc="If ISBN-13 is provided, it has 13 digits; if multiple formats are provided, ISBNs are not reused across formats.",
        parent=pub_node,
        critical=False
    )
    isbn_refs_leaf = evaluator.add_leaf(
        id="Book_3_PublicationISBN_Refs",
        desc="Provide reference URL(s) confirming publication date and/or ISBN information (if provided).",
        parent=pub_node,
        critical=False
    )
    await evaluator.verify(
        claim="The provided reference URL(s) confirm the publication date and/or ISBN information for 'Orbital'.",
        node=isbn_refs_leaf,
        sources=(book.refs_publication + book.refs_isbn) if (book.refs_publication or book.refs_isbn) else None,
        additional_instruction="Verify publication date and/or ISBN via publisher pages or authoritative catalogs."
    )

    return book_node


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 2024–2025 literary awards task.
    Builds a verification tree covering global requirements and per-book checks.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root-level parallel aggregation
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
    # Note: initialize() sets root as non-critical on purpose to allow partial scoring while still gating
    # via critical children, avoiding the "critical children must be all critical" constraint at the root.

    # Extract structured data from the answer
    extraction: AnswerExtraction = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=AnswerExtraction,
        extraction_name="books_extraction"
    )

    # Add ground truth information for transparency
    evaluator.add_ground_truth({
        "expected_winners": EXPECTED_WINNERS,
        "expected_big_five": EXPECTED_BIG_FIVE
    }, gt_type="expected_info")

    # Build global requirements node
    await build_global_requirements(evaluator, root, extraction, answer)

    # Build per-book verification subtrees
    await verify_book_1_james(evaluator, root, extraction)
    await verify_book_2_night_watch(evaluator, root, extraction)
    await verify_book_3_orbital(evaluator, root, extraction)

    # Return summary
    return evaluator.get_summary()