import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# ============================================================================ #
# Task constants                                                               #
# ============================================================================ #
TASK_ID = "california_reads_together_2025_2026"
TASK_DESCRIPTION = """A California-based library consortium is launching a statewide 'California Reads Together' initiative for the 2025-2026 program year and needs to curate a reading list for multiple program tracks. Your task is to identify exactly four books—one for each of the following tracks:

1. Adult Fiction Track: Must be a book that won the National Book Award for Fiction, the Pulitzer Prize for Fiction, OR was selected for Oprah's Book Club between 2020-2024, and was published between 2019-2024.

2. Young Adult Literature Track: Must be a book that received major recognition or a literary award between 2020-2024.

3. Children's Books Track: Must be a book that won the Newbery Medal (not Honor) between 2020-2024, and was published between 2020-2024.

4. Memoir/Nonfiction Track: Must be a memoir or nonfiction book that was featured in either Oprah's Book Club or Reese's Book Club between 2022-2024.

Additional Requirements:
- All four books must be by four different authors (no author can appear twice).
- At least 2 of the 4 selected books must have a confirmed film or television adaptation (announced, in production, or released).
- All books must be currently available in all three formats: print (hardcover or paperback), ebook, and audiobook.

For each book, provide:
- Book title
- Author name
- Publication year
- Publisher
- The specific award won or book club that selected it (with the year)
- If applicable: adaptation status, including production company/streaming service and current status
- Reference URLs verifying: (a) the award/book club selection, (b) publication details, (c) adaptation status (if applicable), and (d) format availability
"""

# Canonical track IDs
ADULT_TRACK = "adult_fiction"
YA_TRACK = "young_adult"
CHILDREN_TRACK = "children"
MEMOIR_TRACK = "memoir_nonfiction"
REQUIRED_TRACKS = [ADULT_TRACK, YA_TRACK, CHILDREN_TRACK, MEMOIR_TRACK]

# ============================================================================ #
# Data models                                                                  #
# ============================================================================ #
class FormatLinks(BaseModel):
    print_url: Optional[str] = None
    ebook_url: Optional[str] = None
    audiobook_url: Optional[str] = None


class Qualification(BaseModel):
    kind: Optional[str] = None  # "award", "book_club", or "recognition"
    name: Optional[str] = None  # e.g., "Pulitzer Prize for Fiction", "Oprah's Book Club"
    year: Optional[str] = None  # e.g., "2023"
    category: Optional[str] = None  # e.g., "Fiction", "Newbery Medal"
    urls: List[str] = Field(default_factory=list)


class BookSelection(BaseModel):
    # Identification
    track: Optional[str] = None  # one of REQUIRED_TRACKS
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    publication_year: Optional[str] = None
    publisher: Optional[str] = None
    category_label: Optional[str] = None  # e.g., "Adult Fiction", "Young Adult", "Children's", "Memoir|Nonfiction"

    # Qualification / Eligibility
    qualification: Optional[Qualification] = None
    qualification_urls: List[str] = Field(default_factory=list)  # redundant helper

    # Publication verification URLs (publisher catalog page, bookseller page, ISBN landing page)
    publication_urls: List[str] = Field(default_factory=list)

    # Format availability
    format_links: FormatLinks = Field(default_factory=FormatLinks)
    format_urls: List[str] = Field(default_factory=list)  # any additional format availability pages

    # Major library distributor availability (e.g., OverDrive/Libby, Hoopla, cloudLibrary, Baker & Taylor, Ingram)
    distributor_urls: List[str] = Field(default_factory=list)

    # Adaptation (optional)
    adaptation_status: Optional[str] = None  # e.g., "announced", "in production", "released"
    adaptation_company: Optional[str] = None  # production company or streaming service
    adaptation_urls: List[str] = Field(default_factory=list)


class ReadingListExtraction(BaseModel):
    books: List[BookSelection] = Field(default_factory=list)


# ============================================================================ #
# Extraction prompt                                                            #
# ============================================================================ #
def prompt_extract_reading_list() -> str:
    return """
Extract exactly FOUR primary book selections from the answer, mapping one to each canonical track:
- adult_fiction
- young_adult
- children
- memoir_nonfiction

Return a JSON with a field "books": an array of four objects, each containing the following fields.
For each book include:
- track: one of "adult_fiction", "young_adult", "children", "memoir_nonfiction"
- title
- authors: array of author names (split co-authors)
- publication_year: 4-digit year string
- publisher
- category_label: short label for category (e.g., "Adult Fiction", "Young Adult", "Children's", "Memoir", "Nonfiction")
- qualification: object capturing eligibility/recognition info with:
    - kind: "award" | "book_club" | "recognition"
    - name: e.g., "Pulitzer Prize for Fiction", "National Book Award for Fiction", "Oprah's Book Club", "Reese's Book Club", "Michael L. Printz Award", etc.
    - year: the year of the award/recognition/club selection
    - category: if applicable (e.g., "Fiction", "Newbery Medal")
    - urls: array of URLs explicitly cited that verify the award/recognition or book-club selection
- qualification_urls: array of the same URLs as qualification.urls for convenience (repeat them here)
- publication_urls: array of URLs that verify publication year and publisher (publisher site, ISBN page, verified retailer or bibliographic database)
- format_links: object with:
    - print_url: URL that shows print (hardcover or paperback) availability
    - ebook_url: URL that shows ebook availability
    - audiobook_url: URL that shows audiobook availability
- format_urls: array of additional URLs (optional) that demonstrate format availability
- distributor_urls: array of URLs that show library acquisition availability via major distributors (e.g., OverDrive/Libby, Hoopla, cloudLibrary, Baker & Taylor, Ingram)
- adaptation_status: if adaptation is claimed, the status such as "announced", "in production", or "released" (else null)
- adaptation_company: name of involved production company or streaming service, if known
- adaptation_urls: array of URLs verifying the adaptation (if claimed)

Rules:
- Extract ONLY what is explicitly present in the answer. Do not invent URLs.
- Use full absolute URLs. If a URL is missing a protocol, prepend http://
- If a particular URL category is not provided in the answer, leave the field empty or null as appropriate.
- Ensure each of the four tracks is represented exactly once. Do not include more than four items.
"""


# ============================================================================ #
# Helpers                                                                      #
# ============================================================================ #
def parse_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(19|20)\d{2}", str(text))
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def year_in_range(year: Optional[int], low: int, high: int) -> bool:
    return year is not None and low <= year <= high


def canonicalize_track(track: Optional[str]) -> Optional[str]:
    if not track:
        return None
    t = track.strip().lower()
    if t in {"adult", "adult fiction", "fiction", "adult_fiction"}:
        return ADULT_TRACK
    if t in {"ya", "young adult", "young_adult"}:
        return YA_TRACK
    if t in {"children", "children's", "kids", "childrens", "children_books", "children_book"}:
        return CHILDREN_TRACK
    if t in {"memoir", "nonfiction", "memoir/nonfiction", "memoir_nonfiction"}:
        return MEMOIR_TRACK
    return track  # assume already canonical


def gather_non_empty(urls: List[Optional[str]]) -> List[str]:
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


def format_sources(selection: BookSelection) -> List[str]:
    fl = selection.format_links
    fmt_urls = gather_non_empty([fl.print_url, fl.ebook_url, fl.audiobook_url]) + selection.format_urls
    return list(dict.fromkeys(fmt_urls))  # dedup, preserve order


def category_sources(selection: BookSelection) -> List[str]:
    # Prefer publication_urls, fall back to qualification urls or format urls
    srcs = list(selection.publication_urls)
    if not srcs:
        if selection.qualification and selection.qualification.urls:
            srcs = list(selection.qualification.urls)
        elif selection.qualification_urls:
            srcs = list(selection.qualification_urls)
        else:
            srcs = list(format_sources(selection))
    return srcs


def qualification_urls(selection: BookSelection) -> List[str]:
    if selection.qualification and selection.qualification.urls:
        return list(selection.qualification.urls)
    return list(selection.qualification_urls)


def count_unique_authors(selections: List[BookSelection]) -> Tuple[int, Dict[str, int]]:
    freq: Dict[str, int] = {}
    for sel in selections:
        for a in sel.authors or []:
            norm = a.strip().lower()
            if norm:
                freq[norm] = freq.get(norm, 0) + 1
    unique_count = sum(1 for _a, c in freq.items() if c == 1)
    return unique_count, freq


def authors_are_unique(selections: List[BookSelection]) -> bool:
    _, freq = count_unique_authors(selections)
    return all(c == 1 for c in freq.values()) if freq else False


def get_books_by_track(extracted: ReadingListExtraction) -> Dict[str, BookSelection]:
    result: Dict[str, BookSelection] = {}
    for b in extracted.books:
        ct = canonicalize_track(b.track)
        if ct and ct not in result:
            # take the first occurrence per track
            if ct != b.track:
                b.track = ct
            result[ct] = b
    return result


def title_and_authors(selection: BookSelection) -> str:
    title = selection.title or "Unknown Title"
    if selection.authors:
        return f"'{title}' by {', '.join(selection.authors)}"
    return f"'{title}'"


# ============================================================================ #
# Verification subroutines for tracks                                          #
# ============================================================================ #
async def verify_adult_selection(evaluator: Evaluator, parent, sel: BookSelection) -> None:
    node = evaluator.add_parallel(
        id="Adult_Fiction_Selection",
        desc="Adult Fiction track book satisfies eligibility and required documentation.",
        parent=parent,
        critical=True,
    )

    # Bibliographic fields presence
    biblio_ok = bool(sel.title and (sel.authors and len(sel.authors) > 0) and sel.publication_year and sel.publisher)
    evaluator.add_custom_node(
        result=biblio_ok,
        id="Adult_Bibliographic_Fields",
        desc="Provides title, author name(s), publication year, and publisher for the Adult selection.",
        parent=node,
        critical=True,
    )

    # Category check (Adult Fiction)
    cat_leaf = evaluator.add_leaf(
        id="Adult_Category",
        desc="The selected work is adult fiction (not YA/children; not memoir/nonfiction).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The book {title_and_authors(sel)} is an adult fiction work (a novel or story collection), not YA/children and not memoir/nonfiction.",
        node=cat_leaf,
        sources=category_sources(sel),
        additional_instruction="Use evidence from the provided page(s). 'Novel' or 'fiction' indicates adult fiction; reject if it is explicitly described as YA, children's, memoir, or nonfiction."
    )

    # Eligibility: NBA Fiction winner OR Pulitzer Fiction winner OR Oprah's Book Club (2020–2024)
    elig_urls = qualification_urls(sel)
    elig_leaf = evaluator.add_leaf(
        id="Adult_Eligibility",
        desc="Meets at least one: (a) won National Book Award for Fiction, OR (b) won Pulitzer Prize for Fiction, OR (c) selected for Oprah's Book Club (2020–2024).",
        parent=node,
        critical=True,
    )
    qname = (sel.qualification.name if sel.qualification else "") or ""
    qyear = (sel.qualification.year if sel.qualification else "") or ""
    claim_text = f"The book {title_and_authors(sel)} qualifies by its recognition: {qname} ({qyear})."
    add_ins = ("Verify that the recognition is one of: National Book Award for Fiction, Pulitzer Prize for Fiction, "
               "or Oprah's Book Club. If it is Oprah's Book Club, ensure selection year is between 2020 and 2024 inclusive.")
    await evaluator.verify(
        claim=claim_text,
        node=elig_leaf,
        sources=elig_urls,
        additional_instruction=add_ins
    )

    # Year-range constraints
    pub_year = parse_year(sel.publication_year)
    evaluator.add_custom_node(
        result=year_in_range(pub_year, 2019, 2024),
        id="Adult_Publication_Year_Range",
        desc="Publication year is between 2019–2024 (inclusive).",
        parent=node,
        critical=True,
    )

    # Required reference URLs presence (award/club, publication, formats)
    fl = sel.format_links
    required_refs_ok = (len(elig_urls) > 0) and (len(sel.publication_urls) > 0) and all(
        [bool(fl.print_url), bool(fl.ebook_url), bool(fl.audiobook_url)]
    )
    evaluator.add_custom_node(
        result=required_refs_ok,
        id="Adult_Required_Reference_URLs",
        desc="Provides reference URLs verifying: (a) the qualifying award/book-club selection, (b) publication details, and (c) format availability.",
        parent=node,
        critical=True,
    )

    # Formats availability
    fmt_leaf = evaluator.add_leaf(
        id="Adult_Format_Availability",
        desc="Adult selection is currently available in all three formats: print (hardcover or paperback), ebook, and audiobook.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The book {title_and_authors(sel)} is currently available in print (hardcover or paperback), ebook, and audiobook formats.",
        node=fmt_leaf,
        sources=format_sources(sel),
        additional_instruction="Confirm explicit availability of each format on the provided URLs. 'Print' can be hardcover or paperback."
    )

    # Major distributor availability (library acquisition)
    dist_present = len(sel.distributor_urls) > 0
    evaluator.add_custom_node(
        result=dist_present,
        id="Adult_Major_Distributor_URLs_Provided",
        desc="Evidence URLs for major distributor availability are provided.",
        parent=node,
        critical=True,
    )
    dist_leaf = evaluator.add_leaf(
        id="Adult_Major_Distributor_Availability",
        desc="Provides verifiable evidence the Adult selection is currently available for library acquisition through major distributors.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The book {title_and_authors(sel)} is available for library acquisition via at least one major distributor.",
        node=dist_leaf,
        sources=sel.distributor_urls,
        additional_instruction="Accept credible distributor/library platforms such as OverDrive/Libby, Hoopla, cloudLibrary, Baker & Taylor, Ingram, or a publisher's library-availability page."
    )


async def verify_ya_selection(evaluator: Evaluator, parent, sel: BookSelection) -> None:
    node = evaluator.add_parallel(
        id="Young_Adult_Selection",
        desc="Young Adult track book satisfies eligibility and required documentation.",
        parent=parent,
        critical=True,
    )

    # Bibliographic fields presence
    biblio_ok = bool(sel.title and (sel.authors and len(sel.authors) > 0) and sel.publication_year and sel.publisher)
    evaluator.add_custom_node(
        result=biblio_ok,
        id="YA_Bibliographic_Fields",
        desc="Provides title, author name(s), publication year, and publisher for the YA selection.",
        parent=node,
        critical=True,
    )

    # Category check (YA)
    cat_leaf = evaluator.add_leaf(
        id="YA_Category",
        desc="The selected work is a Young Adult (YA) book.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The book {title_and_authors(sel)} is categorized as a Young Adult (YA) book.",
        node=cat_leaf,
        sources=category_sources(sel),
        additional_instruction="Verify it is explicitly YA. Reject if labeled children’s, adult, or non-YA non-fiction."
    )

    # Eligibility (major recognition/award 2020–2024)
    elig_urls = qualification_urls(sel)
    elig_leaf = evaluator.add_leaf(
        id="YA_Eligibility",
        desc="Received major recognition or a literary award with year between 2020–2024 (inclusive).",
        parent=node,
        critical=True,
    )
    qname = (sel.qualification.name if sel.qualification else "") or ""
    qyear = (sel.qualification.year if sel.qualification else "") or ""
    await evaluator.verify(
        claim=f"The YA book {title_and_authors(sel)} received major recognition or a literary award in {qyear}: {qname}.",
        node=elig_leaf,
        sources=elig_urls,
        additional_instruction="Confirm that the recognition or award is significant (e.g., Printz Award, National Book Award for Young People's Literature, etc.), with the year between 2020 and 2024 inclusive."
    )
    # Year range custom check
    qy_int = parse_year(qyear)
    evaluator.add_custom_node(
        result=year_in_range(qy_int, 2020, 2024),
        id="YA_Eligibility_Year_Range",
        desc="YA recognition/award year is between 2020–2024 (inclusive).",
        parent=node,
        critical=True,
    )

    # Required reference URLs presence
    fl = sel.format_links
    required_refs_ok = (len(elig_urls) > 0) and (len(sel.publication_urls) > 0) and all(
        [bool(fl.print_url), bool(fl.ebook_url), bool(fl.audiobook_url)]
    )
    evaluator.add_custom_node(
        result=required_refs_ok,
        id="YA_Required_Reference_URLs",
        desc="Provides reference URLs verifying: (a) the qualifying recognition/award, (b) publication details, and (c) format availability.",
        parent=node,
        critical=True,
    )

    # Formats availability
    fmt_leaf = evaluator.add_leaf(
        id="YA_Format_Availability",
        desc="YA selection is currently available in all three formats: print (hardcover or paperback), ebook, and audiobook.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The book {title_and_authors(sel)} is currently available in print (hardcover or paperback), ebook, and audiobook formats.",
        node=fmt_leaf,
        sources=format_sources(sel),
        additional_instruction="Confirm explicit availability of each format on the provided URLs. 'Print' can be hardcover or paperback."
    )

    # Major distributor availability
    dist_present = len(sel.distributor_urls) > 0
    evaluator.add_custom_node(
        result=dist_present,
        id="YA_Major_Distributor_URLs_Provided",
        desc="Evidence URLs for major distributor availability are provided.",
        parent=node,
        critical=True,
    )
    dist_leaf = evaluator.add_leaf(
        id="YA_Major_Distributor_Availability",
        desc="Provides verifiable evidence the YA selection is currently available for library acquisition through major distributors.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The book {title_and_authors(sel)} is available for library acquisition via at least one major distributor.",
        node=dist_leaf,
        sources=sel.distributor_urls,
        additional_instruction="Accept credible distributor/library platforms such as OverDrive/Libby, Hoopla, cloudLibrary, Baker & Taylor, Ingram, or a publisher's library-availability page."
    )


async def verify_children_selection(evaluator: Evaluator, parent, sel: BookSelection) -> None:
    node = evaluator.add_parallel(
        id="Children_Selection",
        desc="Children's track book satisfies eligibility and required documentation.",
        parent=parent,
        critical=True,
    )

    # Bibliographic presence
    biblio_ok = bool(sel.title and (sel.authors and len(sel.authors) > 0) and sel.publication_year and sel.publisher)
    evaluator.add_custom_node(
        result=biblio_ok,
        id="Children_Bibliographic_Fields",
        desc="Provides title, author name(s), publication year, and publisher for the Children's selection.",
        parent=node,
        critical=True,
    )

    # Category: Children's
    cat_leaf = evaluator.add_leaf(
        id="Children_Category",
        desc="The selected work is a children's book.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The book {title_and_authors(sel)} is a children's book (suitable for middle grade/children).",
        node=cat_leaf,
        sources=category_sources(sel),
        additional_instruction="Look for explicit children's categorization, middle grade indications, publisher age range, etc."
    )

    # Eligibility: Newbery Medal (not Honor) 2020–2024
    elig_urls = qualification_urls(sel)
    elig_leaf = evaluator.add_leaf(
        id="Children_Eligibility",
        desc="Won the Newbery Medal (not Honor) with award year between 2020–2024 (inclusive).",
        parent=node,
        critical=True,
    )
    qname = (sel.qualification.name if sel.qualification else "") or ""
    qyear = (sel.qualification.year if sel.qualification else "") or ""
    await evaluator.verify(
        claim=f"The book {title_and_authors(sel)} won the Newbery Medal (not Honor) in {qyear}.",
        node=elig_leaf,
        sources=elig_urls,
        additional_instruction="Confirm that it is the Newbery Medal winner (not 'Honor'), and that the award year is between 2020 and 2024 inclusive."
    )
    # Year-range custom
    qy_int = parse_year(qyear)
    evaluator.add_custom_node(
        result=year_in_range(qy_int, 2020, 2024),
        id="Children_Eligibility_Year_Range",
        desc="Newbery Medal award year is between 2020–2024 (inclusive).",
        parent=node,
        critical=True,
    )

    # Publication year 2020–2024
    pub_year = parse_year(sel.publication_year)
    evaluator.add_custom_node(
        result=year_in_range(pub_year, 2020, 2024),
        id="Children_Publication_Year_Range",
        desc="Publication year is between 2020–2024 (inclusive).",
        parent=node,
        critical=True,
    )

    # Required reference URLs presence
    fl = sel.format_links
    required_refs_ok = (len(elig_urls) > 0) and (len(sel.publication_urls) > 0) and all(
        [bool(fl.print_url), bool(fl.ebook_url), bool(fl.audiobook_url)]
    )
    evaluator.add_custom_node(
        result=required_refs_ok,
        id="Children_Required_Reference_URLs",
        desc="Provides reference URLs verifying: (a) Newbery Medal win, (b) publication details, and (c) format availability.",
        parent=node,
        critical=True,
    )

    # Formats availability
    fmt_leaf = evaluator.add_leaf(
        id="Children_Format_Availability",
        desc="Children's selection is currently available in all three formats: print (hardcover or paperback), ebook, and audiobook.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The book {title_and_authors(sel)} is currently available in print (hardcover or paperback), ebook, and audiobook formats.",
        node=fmt_leaf,
        sources=format_sources(sel),
        additional_instruction="Confirm explicit availability of each format on the provided URLs. 'Print' can be hardcover or paperback."
    )

    # Major distributor availability
    dist_present = len(sel.distributor_urls) > 0
    evaluator.add_custom_node(
        result=dist_present,
        id="Children_Major_Distributor_URLs_Provided",
        desc="Evidence URLs for major distributor availability are provided.",
        parent=node,
        critical=True,
    )
    dist_leaf = evaluator.add_leaf(
        id="Children_Major_Distributor_Availability",
        desc="Provides verifiable evidence the Children's selection is currently available for library acquisition through major distributors.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The book {title_and_authors(sel)} is available for library acquisition via at least one major distributor.",
        node=dist_leaf,
        sources=sel.distributor_urls,
        additional_instruction="Accept credible distributor/library platforms such as OverDrive/Libby, Hoopla, cloudLibrary, Baker & Taylor, Ingram, or a publisher's library-availability page."
    )


async def verify_memoir_selection(evaluator: Evaluator, parent, sel: BookSelection) -> None:
    node = evaluator.add_parallel(
        id="Memoir_Nonfiction_Selection",
        desc="Memoir/Nonfiction track book satisfies eligibility and required documentation.",
        parent=parent,
        critical=True,
    )

    # Bibliographic presence
    biblio_ok = bool(sel.title and (sel.authors and len(sel.authors) > 0) and sel.publication_year and sel.publisher)
    evaluator.add_custom_node(
        result=biblio_ok,
        id="Memoir_Bibliographic_Fields",
        desc="Provides title, author name(s), publication year, and publisher for the Memoir/Nonfiction selection.",
        parent=node,
        critical=True,
    )

    # Category: Memoir/Nonfiction
    cat_leaf = evaluator.add_leaf(
        id="Memoir_Category",
        desc="The selected work is memoir or nonfiction (not fiction/YA/children).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The book {title_and_authors(sel)} is a memoir or nonfiction book (not fiction/YA/children).",
        node=cat_leaf,
        sources=category_sources(sel),
        additional_instruction="Confirm non-fiction or memoir categorization; reject if described as a novel/fiction or YA/children."
    )

    # Eligibility: Oprah's or Reese's Book Club 2022–2024
    elig_urls = qualification_urls(sel)
    elig_leaf = evaluator.add_leaf(
        id="Memoir_Eligibility",
        desc="Featured in Oprah's Book Club or Reese's Book Club with selection year between 2022–2024 (inclusive).",
        parent=node,
        critical=True,
    )
    qname = (sel.qualification.name if sel.qualification else "") or ""
    qyear = (sel.qualification.year if sel.qualification else "") or ""
    await evaluator.verify(
        claim=f"The book {title_and_authors(sel)} was featured in {qname} in {qyear}.",
        node=elig_leaf,
        sources=elig_urls,
        additional_instruction="Confirm feature in Oprah's Book Club or Reese's Book Club, with selection year between 2022 and 2024 inclusive."
    )
    qy_int = parse_year(qyear)
    evaluator.add_custom_node(
        result=year_in_range(qy_int, 2022, 2024),
        id="Memoir_Eligibility_Year_Range",
        desc="Book-club feature year is between 2022–2024 (inclusive).",
        parent=node,
        critical=True,
    )

    # Required reference URLs presence
    fl = sel.format_links
    required_refs_ok = (len(elig_urls) > 0) and (len(sel.publication_urls) > 0) and all(
        [bool(fl.print_url), bool(fl.ebook_url), bool(fl.audiobook_url)]
    )
    evaluator.add_custom_node(
        result=required_refs_ok,
        id="Memoir_Required_Reference_URLs",
        desc="Provides reference URLs verifying: (a) book-club feature, (b) publication details, and (c) format availability.",
        parent=node,
        critical=True,
    )

    # Formats availability
    fmt_leaf = evaluator.add_leaf(
        id="Memoir_Format_Availability",
        desc="Memoir/Nonfiction selection is currently available in all three formats: print (hardcover or paperback), ebook, and audiobook.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The book {title_and_authors(sel)} is currently available in print (hardcover or paperback), ebook, and audiobook formats.",
        node=fmt_leaf,
        sources=format_sources(sel),
        additional_instruction="Confirm explicit availability of each format on the provided URLs. 'Print' can be hardcover or paperback."
    )

    # Major distributor availability
    dist_present = len(sel.distributor_urls) > 0
    evaluator.add_custom_node(
        result=dist_present,
        id="Memoir_Major_Distributor_URLs_Provided",
        desc="Evidence URLs for major distributor availability are provided.",
        parent=node,
        critical=True,
    )
    dist_leaf = evaluator.add_leaf(
        id="Memoir_Major_Distributor_Availability",
        desc="Provides verifiable evidence the Memoir/Nonfiction selection is currently available for library acquisition through major distributors.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The book {title_and_authors(sel)} is available for library acquisition via at least one major distributor.",
        node=dist_leaf,
        sources=sel.distributor_urls,
        additional_instruction="Accept credible distributor/library platforms such as OverDrive/Libby, Hoopla, cloudLibrary, Baker & Taylor, Ingram, or a publisher's library-availability page."
    )


# ============================================================================ #
# Cross-selection checks                                                       #
# ============================================================================ #
async def verify_minimum_adaptations_offtree(evaluator: Evaluator, selections: List[BookSelection]) -> Tuple[int, Dict[str, bool]]:
    """
    Verify adaptation claims off-tree (no nodes), returning count of verified adaptations and per-track results.
    """
    results: Dict[str, bool] = {}

    async def _check_one(sel: BookSelection) -> Tuple[str, bool]:
        track = sel.track or "unknown_track"
        # Consider claimed only if there is at least one adaptation URL and a status string
        claimed = bool(sel.adaptation_status and len(sel.adaptation_urls) > 0)
        if not claimed:
            return track, False
        claim = f"The book {title_and_authors(sel)} has a confirmed film or television adaptation (announced, in production, or released)."
        add_ins = "Verify from the provided URL(s) that an adaptation is real/confirmed (trade press, official announcements, major outlets)."
        ok = await evaluator.verify(claim=claim, node=None, sources=sel.adaptation_urls, additional_instruction=add_ins)
        return track, ok

    tasks = [asyncio.create_task(_check_one(sel)) for sel in selections]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    verified_count = 0
    for outcome in outcomes:
        if isinstance(outcome, Exception):
            continue
        track, ok = outcome
        results[track] = ok
        if ok:
            verified_count += 1

    return verified_count, results


# ============================================================================ #
# Main evaluation entry point                                                  #
# ============================================================================ #
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured reading list
    extracted: ReadingListExtraction = await evaluator.extract(
        prompt=prompt_extract_reading_list(),
        template_class=ReadingListExtraction,
        extraction_name="reading_list_extraction",
    )

    # Create a critical top-level node
    top = evaluator.add_sequential(
        id="California_Reads_Initiative",
        desc="Evaluate the complete 4-book reading list (one per track) against all stated constraints.",
        parent=root,
        critical=True,
    )

    # 1) Response structure
    resp_node = evaluator.add_parallel(
        id="Response_Structure",
        desc="Provides exactly 4 total primary selections: exactly one for each track (Adult Fiction, Young Adult, Children's, Memoir/Nonfiction) and no extra primary book selections.",
        parent=top,
        critical=True,
    )

    # Normalize track mapping
    books_by_track = get_books_by_track(extracted)
    # Count structure checks
    count_is_four = len(extracted.books) == 4
    evaluator.add_custom_node(
        result=count_is_four,
        id="Exactly_Four_Selections",
        desc="Exactly 4 total primary selections are provided.",
        parent=resp_node,
        critical=True,
    )

    tracks_ok = all(t in books_by_track for t in REQUIRED_TRACKS) and len(books_by_track) == 4
    evaluator.add_custom_node(
        result=tracks_ok,
        id="One_Per_Track",
        desc="Exactly one selection for each required track is provided.",
        parent=resp_node,
        critical=True,
    )

    # 2) Track selections (parallel critical)
    tracks_parent = evaluator.add_parallel(
        id="Track_Selections",
        desc="Each track’s selected book meets its track-specific constraints and required metadata is provided.",
        parent=top,
        critical=True,
    )

    # Provide placeholder selections if missing to keep tree shape deterministic
    adult_sel = books_by_track.get(ADULT_TRACK, BookSelection(track=ADULT_TRACK))
    ya_sel = books_by_track.get(YA_TRACK, BookSelection(track=YA_TRACK))
    children_sel = books_by_track.get(CHILDREN_TRACK, BookSelection(track=CHILDREN_TRACK))
    memoir_sel = books_by_track.get(MEMOIR_TRACK, BookSelection(track=MEMOIR_TRACK))

    await verify_adult_selection(evaluator, tracks_parent, adult_sel)
    await verify_ya_selection(evaluator, tracks_parent, ya_sel)
    await verify_children_selection(evaluator, tracks_parent, children_sel)
    await verify_memoir_selection(evaluator, tracks_parent, memoir_sel)

    # 3) Cross-selection requirements
    cross = evaluator.add_parallel(
        id="Cross_Selection_Requirements",
        desc="Constraints that depend on the set of all four selections.",
        parent=top,
        critical=True,
    )

    # Unique authors across all four books (consider all co-authors)
    uniq_ok = authors_are_unique([adult_sel, ya_sel, children_sel, memoir_sel])
    evaluator.add_custom_node(
        result=uniq_ok,
        id="Unique_Authors",
        desc="No author appears more than once across the four selected books (considering all listed co-authors).",
        parent=cross,
        critical=True,
    )

    # Minimum adaptations: at least 2 verified
    verified_count, adapt_results = await verify_minimum_adaptations_offtree(
        evaluator, [adult_sel, ya_sel, children_sel, memoir_sel]
    )
    evaluator.add_custom_node(
        result=(verified_count >= 2),
        id="Minimum_Adaptations",
        desc="At least 2 of the 4 selected books have a confirmed film/TV adaptation with verifying URL(s).",
        parent=cross,
        critical=True,
    )
    evaluator.add_custom_info(
        info={
            "verified_adaptation_count": verified_count,
            "per_track_results": adapt_results,
        },
        info_type="cross_selection_analysis",
        info_name="adaptation_verification_summary"
    )

    return evaluator.get_summary()