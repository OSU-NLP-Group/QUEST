import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_2025_award_winning_books"
TASK_DESCRIPTION = """
Identify three books that won major literary awards in the United States in 2025, with each book winning a different award from the following categories: (1) the Pulitzer Prize for Fiction, (2) the National Book Award for Fiction, and (3) The Center for Fiction First Novel Prize. For each award-winning book, provide: book details (title, author, publisher, month & year of U.S. publication, 13‑digit ISBN), award information (official award name and confirmation that it was won in 2025 with authoritative URL), author information (current professional affiliation or U.S. city/state of residence with authoritative URL), and at least one additional major recognition with URL. All information must be supported by authoritative URLs (official org sites, publishers, universities, or major news outlets such as NPR, NYT, or Publishers Weekly).
"""

AWARD_YEAR = "2025"
EXPECTED_AWARDS = {
    "pulitzer": "Pulitzer Prize for Fiction",
    "nba": "National Book Award for Fiction",
    "cff": "The Center for Fiction First Novel Prize",
}

CATEGORY_DESCRIPTIONS = {
    "pulitzer": {
        "winner": "Book that won the 2025 Pulitzer Prize for Fiction with complete information",
        "book_details": "Complete bibliographic information for the Pulitzer Prize-winning book",
        "award_info": "Award information confirming the 2025 Pulitzer Prize for Fiction win",
        "author_info": "Author's professional affiliation or U.S. residence from official sources",
        "additional_recognition": "At least one other major literary recognition received by the book",
        "title": "Complete book title is provided",
        "author": "Author's full name is provided",
        "publisher": "Publisher name is provided",
        "pub_date": "Publication month and year are provided",
        "isbn": "13-digit ISBN for U.S. print edition is provided",
        "book_reference": "Reference URL from authoritative source confirming book details",
        "award_name": "Full official award name (Pulitzer Prize for Fiction) is stated",
        "award_year": "Award year is confirmed as 2025",
        "award_reference": "Reference URL from official Pulitzer Prize website or major news outlet confirming the win",
        "author_affiliation": "Author's current professional affiliation or U.S. city and state of residence is provided",
        "author_reference": "Reference URL from authoritative source confirming author information",
        "other_recognition": "Name of additional major award, finalist designation, or best books list is provided",
        "recognition_reference": "Reference URL confirming the additional recognition",
    },
    "nba": {
        "winner": "Book that won the 2025 National Book Award for Fiction with complete information",
        "book_details": "Complete bibliographic information for the National Book Award-winning book",
        "award_info": "Award information confirming the 2025 National Book Award for Fiction win",
        "author_info": "Author's professional affiliation or U.S. residence from official sources",
        "additional_recognition": "At least one other major literary recognition received by the book",
        "title": "Complete book title is provided",
        "author": "Author's full name is provided",
        "publisher": "Publisher name is provided",
        "pub_date": "Publication month and year are provided",
        "isbn": "13-digit ISBN for U.S. print edition is provided",
        "book_reference": "Reference URL from authoritative source confirming book details",
        "award_name": "Full official award name (National Book Award for Fiction) is stated",
        "award_year": "Award year is confirmed as 2025",
        "award_reference": "Reference URL from official National Book Foundation website or major news outlet confirming the win",
        "author_affiliation": "Author's current professional affiliation or U.S. city and state of residence is provided",
        "author_reference": "Reference URL from authoritative source confirming author information",
        "other_recognition": "Name of additional major award, finalist designation, or best books list is provided",
        "recognition_reference": "Reference URL confirming the additional recognition",
    },
    "cff": {
        "winner": "Book that won the 2025 Center for Fiction First Novel Prize with complete information",
        "book_details": "Complete bibliographic information for the Center for Fiction First Novel Prize-winning book",
        "award_info": "Award information confirming the 2025 Center for Fiction First Novel Prize win",
        "author_info": "Author's professional affiliation or U.S. residence from official sources",
        "additional_recognition": "At least one other major literary recognition received by the book",
        "title": "Complete book title is provided",
        "author": "Author's full name is provided",
        "publisher": "Publisher name is provided",
        "pub_date": "Publication month and year are provided",
        "isbn": "13-digit ISBN for U.S. print edition is provided",
        "book_reference": "Reference URL from authoritative source confirming book details",
        "award_name": "Full official award name (The Center for Fiction First Novel Prize) is stated",
        "award_year": "Award year is confirmed as 2025",
        "award_reference": "Reference URL from official Center for Fiction website or major news outlet confirming the win",
        "author_affiliation": "Author's current professional affiliation or U.S. city and state of residence is provided",
        "author_reference": "Reference URL from authoritative source confirming author information",
        "other_recognition": "Name of additional major award, finalist designation, or best books list is provided",
        "recognition_reference": "Reference URL confirming the additional recognition",
    },
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BookDetails(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_month_year: Optional[str] = None
    isbn13: Optional[str] = None
    book_reference_urls: List[str] = Field(default_factory=list)


class AwardDetails(BaseModel):
    award_name: Optional[str] = None
    award_year: Optional[str] = None
    award_reference_urls: List[str] = Field(default_factory=list)


class AuthorDetails(BaseModel):
    affiliation: Optional[str] = None  # e.g., "Associate Professor at XYZ University" or "Writer-in-Residence at ..."
    residence_city_state: Optional[str] = None  # e.g., "Brooklyn, NY"
    author_reference_urls: List[str] = Field(default_factory=list)


class RecognitionDetails(BaseModel):
    recognition_name: Optional[str] = None  # e.g., "NYT Notable Books of 2025", "Finalist, PEN/Faulkner Award"
    recognition_reference_urls: List[str] = Field(default_factory=list)


class PrizePackage(BaseModel):
    book: Optional[BookDetails] = None
    award: Optional[AwardDetails] = None
    author: Optional[AuthorDetails] = None
    recognition: Optional[RecognitionDetails] = None


class AllWinners(BaseModel):
    pulitzer: Optional[PrizePackage] = None
    nba: Optional[PrizePackage] = None
    cff: Optional[PrizePackage] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all_winners() -> str:
    return """
    Extract structured information from the answer for three different 2025 U.S. literary award winners:
    – Pulitzer Prize for Fiction (2025 winner)
    – National Book Award for Fiction (2025 winner)
    – The Center for Fiction First Novel Prize (2025 winner)

    For each of the three awards, extract a JSON object with keys: "book", "award", "author", and "recognition".

    For the "book" object, extract:
    - title: the complete title of the winning book (string)
    - author: the full author name (string)
    - publisher: the publisher name (string)
    - publication_month_year: the U.S. publication month and year as a single string (e.g., "March 2025"; if a day is also present, keep only month and year)
    - isbn13: the 13-digit ISBN for the U.S. print edition (keep hyphens if present; if multiple ISBNs are listed, choose the 13-digit print ISBN)
    - book_reference_urls: an array of URLs that directly confirm bibliographic details (prefer publisher pages, official imprint pages, ISBN listings, award org pages that list ISBN, or major outlets)

    For the "award" object, extract:
    - award_name: the full official name of the specific award category (e.g., "Pulitzer Prize for Fiction")
    - award_year: the year string (should be "2025")
    - award_reference_urls: an array of authoritative URLs that explicitly confirm the win (prefer the award organization's site; major outlets like NPR, NYT, Publishers Weekly are acceptable)

    For the "author" object, extract:
    - affiliation: the author's current professional affiliation or role, if provided (e.g., "Assistant Professor at ...", "Writer-in-Residence at ...")
    - residence_city_state: the author's current U.S. city and state of residence if provided (e.g., "Brooklyn, NY")
    - author_reference_urls: an array of authoritative URLs (publisher bio, award org bio page, university page) that confirm the affiliation or residence

    For the "recognition" object, extract:
    - recognition_name: at least one other major literary honor or "best books of 2025" style recognition for the same book (e.g., "New York Times Notable Books of 2025", "Finalist, PEN/Faulkner Award")
    - recognition_reference_urls: an array of authoritative URLs confirming that recognition

    Return the final JSON with top-level keys: "pulitzer", "nba", and "cff", each mapping to its object as specified.
    Rules:
    - Extract ONLY what is explicitly present in the answer. If any field is missing, set it to null (for strings) or [] (for URL arrays).
    - Extract only valid URLs that appear in the answer. Include full URLs with http/https.
    - Do not fabricate data; do not infer missing details.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def is_isbn13(s: Optional[str]) -> bool:
    if not non_empty(s):
        return False
    digits = re.sub(r"\D", "", s)  # strip hyphens/spaces
    return len(digits) == 13 and digits.isdigit()


def build_book_details_claim(book: BookDetails) -> str:
    parts = []
    if non_empty(book.title):
        parts.append(f"title '{book.title}'")
    if non_empty(book.author):
        parts.append(f"author {book.author}")
    if non_empty(book.publisher):
        parts.append(f"publisher {book.publisher}")
    if non_empty(book.publication_month_year):
        parts.append(f"U.S. publication date {book.publication_month_year}")
    if non_empty(book.isbn13):
        parts.append(f"13-digit ISBN {book.isbn13}")
    joined = ", ".join(parts) if parts else "the key bibliographic details"
    return f"The referenced page explicitly confirms {joined} for the same book."


async def add_url_verified_leaf(
    evaluator: Evaluator,
    node_id: str,
    desc: str,
    parent,
    critical: bool,
    claim: str,
    urls: List[str],
    additional_instruction: str,
) -> None:
    """Create a verifying leaf when URLs exist; otherwise add a failing custom node."""
    # Clean URLs: remove empties/whitespace
    urls = [u.strip() for u in urls if non_empty(u)]
    if urls:
        leaf = evaluator.add_leaf(id=node_id, desc=desc, parent=parent, critical=critical)
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=additional_instruction,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=desc,
            parent=parent,
            critical=critical,
        )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_book_details(
    evaluator: Evaluator,
    parent,
    prefix: str,
    labels: Dict[str, str],
    book: Optional[BookDetails],
) -> None:
    details_node = evaluator.add_parallel(
        id=f"{prefix}_book_details",
        desc=labels["book_details"],
        parent=parent,
        critical=True,
    )

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=non_empty(book.title) if book else False,
        id=f"{prefix}_title",
        desc=labels["title"],
        parent=details_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=non_empty(book.author) if book else False,
        id=f"{prefix}_author",
        desc=labels["author"],
        parent=details_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=non_empty(book.publisher) if book else False,
        id=f"{prefix}_publisher",
        desc=labels["publisher"],
        parent=details_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=non_empty(book.publication_month_year) if book else False,
        id=f"{prefix}_pub_date",
        desc=labels["pub_date"],
        parent=details_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=is_isbn13(book.isbn13) if book else False,
        id=f"{prefix}_isbn",
        desc=labels["isbn"],
        parent=details_node,
        critical=True,
    )

    # Reference URL verification (critical)
    await add_url_verified_leaf(
        evaluator=evaluator,
        node_id=f"{prefix}_book_reference",
        desc=labels["book_reference"],
        parent=details_node,
        critical=True,
        claim=build_book_details_claim(book or BookDetails()),
        urls=(book.book_reference_urls if book else []),
        additional_instruction=(
            "Judge only by the provided webpage(s). The page must unambiguously refer to the same book and confirm: "
            "title and author; and strongly prefer also publisher, publication month+year (for U.S. release), and the 13‑digit ISBN. "
            "Accept authoritative sources only (publisher/imprint sites, ISBN listings from publishers, official award org pages with ISBNs, "
            "or major news outlets such as NPR, The New York Times, Publishers Weekly). If the evidence is not on the page(s), return Incorrect."
        ),
    )


async def verify_award_info(
    evaluator: Evaluator,
    parent,
    prefix: str,
    labels: Dict[str, str],
    expected_award_name: str,
    award: Optional[AwardDetails],
    book: Optional[BookDetails],
) -> None:
    award_node = evaluator.add_parallel(
        id=f"{prefix}_award_info",
        desc=labels["award_info"],
        parent=parent,
        critical=True,
    )

    # Award name equality (critical simple verify)
    award_name_leaf = evaluator.add_leaf(
        id=f"{prefix}_award_name",
        desc=labels["award_name"],
        parent=award_node,
        critical=True,
    )
    extracted_name = award.award_name if award else ""
    await evaluator.verify(
        claim=f"The award name '{extracted_name}' is equivalent to the official name '{expected_award_name}'.",
        node=award_name_leaf,
        additional_instruction="Allow minor variations in punctuation/casing (e.g., hyphenation, 'The' prefix), but the meaning must be the same award category.",
    )

    # Award year is 2025 (critical simple verify)
    award_year_leaf = evaluator.add_leaf(
        id=f"{prefix}_award_year",
        desc=labels["award_year"],
        parent=award_node,
        critical=True,
    )
    extracted_year = award.award_year if award else ""
    await evaluator.verify(
        claim=f"The extracted award year string '{extracted_year}' indicates the year {AWARD_YEAR}.",
        node=award_year_leaf,
        additional_instruction="Treat '2025' (with or without extra text) as indicating the year 2025; anything else should be Incorrect.",
    )

    # Award reference URL(s) confirming win (critical)
    title = (book.title if book and non_empty(book.title) else "the book")
    author = (book.author if book and non_empty(book.author) else "the author")
    await add_url_verified_leaf(
        evaluator=evaluator,
        node_id=f"{prefix}_award_reference",
        desc=labels["award_reference"],
        parent=award_node,
        critical=True,
        claim=(
            f"The webpage confirms that '{title}' by {author} is the WINNER of the {AWARD_YEAR} {expected_award_name}."
        ),
        urls=(award.award_reference_urls if award else []),
        additional_instruction=(
            "Pass only if the page explicitly shows the book as the 2025 WINNER of the specified award category. "
            "Do NOT accept longlists, shortlists, finalists, nominees, or different years/categories. "
            "Accept official award org pages (e.g., Pulitzer/National Book Foundation/Center for Fiction) or major news outlets (NPR, NYT, Publishers Weekly)."
        ),
    )


async def verify_author_info(
    evaluator: Evaluator,
    parent,
    prefix: str,
    labels: Dict[str, str],
    author_info: Optional[AuthorDetails],
    book: Optional[BookDetails],
) -> None:
    author_node = evaluator.add_parallel(
        id=f"{prefix}_author_info",
        desc=labels["author_info"],
        parent=parent,
        critical=True,
    )

    # Provide either affiliation or residence (critical existence)
    has_aff_or_res = False
    aff_val = ""
    res_val = ""
    if author_info:
        aff_val = author_info.affiliation or ""
        res_val = author_info.residence_city_state or ""
        has_aff_or_res = non_empty(aff_val) or non_empty(res_val)

    evaluator.add_custom_node(
        result=has_aff_or_res,
        id=f"{prefix}_author_affiliation",
        desc=labels["author_affiliation"],
        parent=author_node,
        critical=True,
    )

    # Reference URL(s) confirming author info (critical)
    chosen_claim = ""
    if non_empty(aff_val):
        chosen_claim = (
            f"The page states that {(book.author if book and non_empty(book.author) else 'the author')} currently holds the professional affiliation: '{aff_val}'."
        )
    elif non_empty(res_val):
        chosen_claim = (
            f"The page states that {(book.author if book and non_empty(book.author) else 'the author')} currently resides in {res_val} (U.S.)."
        )
    else:
        chosen_claim = (
            "The page confirms the author's current professional affiliation or current U.S. city and state of residence."
        )

    await add_url_verified_leaf(
        evaluator=evaluator,
        node_id=f"{prefix}_author_reference",
        desc=labels["author_reference"],
        parent=author_node,
        critical=True,
        claim=chosen_claim,
        urls=(author_info.author_reference_urls if author_info else []),
        additional_instruction=(
            "Accept only authoritative sources: publisher author pages, award org bios, or university profiles. "
            "Do not rely on Wikipedia or random blogs. The page must clearly indicate current affiliation or current U.S. residence. "
            "If the page is outdated or ambiguous, return Incorrect."
        ),
    )


async def verify_additional_recognition(
    evaluator: Evaluator,
    parent,
    prefix: str,
    labels: Dict[str, str],
    recognition: Optional[RecognitionDetails],
    book: Optional[BookDetails],
) -> None:
    recog_node = evaluator.add_parallel(
        id=f"{prefix}_additional_recognition",
        desc=labels["additional_recognition"],
        parent=parent,
        critical=True,
    )

    # Recognition name provided (critical existence)
    evaluator.add_custom_node(
        result=non_empty(recognition.recognition_name) if recognition else False,
        id=f"{prefix}_other_recognition",
        desc=labels["other_recognition"],
        parent=recog_node,
        critical=True,
    )

    # Reference URL(s) confirming recognition (critical)
    title = (book.title if book and non_empty(book.title) else "the book")
    author = (book.author if book and non_empty(book.author) else "the author")
    recog_name = (recognition.recognition_name if recognition and non_empty(recognition.recognition_name) else "the stated recognition")
    await add_url_verified_leaf(
        evaluator=evaluator,
        node_id=f"{prefix}_recognition_reference",
        desc=labels["recognition_reference"],
        parent=recog_node,
        critical=True,
        claim=(
            f"The page confirms that '{title}' by {author} received the recognition: {recog_name} "
            f"(e.g., a major award honor/finalist or appearance on a prominent 'best books of 2025' list)."
        ),
        urls=(recognition.recognition_reference_urls if recognition else []),
        additional_instruction=(
            "Accept recognitions that are clearly major (e.g., finalists for notable awards, NYT Notable/Best Books lists, similar caliber outlets). "
            "Ensure the recognition applies to the same book and is for 2025 if the list/recognition is year-specific. Reject unrelated titles or years."
        ),
    )


async def verify_award_package(
    evaluator: Evaluator,
    root,
    category_key: str,
    package: Optional[PrizePackage],
) -> None:
    expected_name = EXPECTED_AWARDS[category_key]
    labels = CATEGORY_DESCRIPTIONS[category_key]

    # Top-level node for this award (non-critical; parallel aggregation)
    main_node = evaluator.add_parallel(
        id=f"{category_key}_winner",
        desc=labels["winner"],
        parent=root,
        critical=False,
    )

    # Unpack sub-objects
    book = package.book if package else None
    award = package.award if package else None
    author_info = package.author if package else None
    recognition = package.recognition if package else None

    # Build 4 critical sections
    await verify_book_details(evaluator, main_node, category_key, labels, book)
    await verify_award_info(evaluator, main_node, category_key, labels, expected_name, award, book)
    await verify_author_info(evaluator, main_node, category_key, labels, author_info, book)
    await verify_additional_recognition(evaluator, main_node, category_key, labels, recognition, book)


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
    # Initialize evaluator (root: parallel aggregation as rubric specifies)
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

    # Extract structured information once
    extracted: AllWinners = await evaluator.extract(
        prompt=prompt_extract_all_winners(),
        template_class=AllWinners,
        extraction_name="award_winners_extraction",
    )

    # Optional: record expected ground truth categories (not specific titles)
    evaluator.add_ground_truth({
        "expected_awards": EXPECTED_AWARDS,
        "expected_year": AWARD_YEAR,
        "notes": "This task requires one distinct 2025 winner for each listed award category."
    }, gt_type="expected_award_categories")

    # Verify each award package independently (parallel under root)
    await verify_award_package(evaluator, root, "pulitzer", extracted.pulitzer)
    await verify_award_package(evaluator, root, "nba", extracted.nba)
    await verify_award_package(evaluator, root, "cff", extracted.cff)

    # Return unified summary
    return evaluator.get_summary()