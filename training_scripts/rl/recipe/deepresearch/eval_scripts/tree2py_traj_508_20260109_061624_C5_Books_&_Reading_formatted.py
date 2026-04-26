import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "novel_award_ca_prof_broadway_nyc_publisher"
TASK_DESCRIPTION = (
    "Identify a novel that won a major U.S. literary fiction award (National Book Award, Pulitzer Prize, or PEN/Faulkner Award) "
    "in either 2023 or 2024, where the author currently holds a professorial position at a university in California, and the book was "
    "published by a publishing house with its headquarters on Broadway in New York City. The novel must have been published between 2022 "
    "and 2024, inclusive. Provide the following information: (1) The book's title, (2) The author's full name, (3) The specific award won, "
    "the year it was won, and the category, (4) The name of the California university where the author currently teaches, the state, and "
    "the author's academic position title, (5) The publisher's name, (6) The publisher's complete headquarters address on Broadway in NYC, "
    "including street address and ZIP code, (7) The book's publication year, and (8) Reference URLs verifying the award win, the author's "
    "university affiliation, and the publisher's headquarters location."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AwardInfo(BaseModel):
    award_name: Optional[str] = None
    award_year: Optional[str] = None
    award_category: Optional[str] = None
    award_urls: List[str] = Field(default_factory=list)


class AffiliationInfo(BaseModel):
    university_name: Optional[str] = None
    university_state: Optional[str] = None
    position_title: Optional[str] = None
    affiliation_urls: List[str] = Field(default_factory=list)


class HQAddress(BaseModel):
    street_address: Optional[str] = None
    city: Optional[str] = None
    zip_code: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)


class PublisherInfo(BaseModel):
    publisher_name: Optional[str] = None
    headquarters: HQAddress = Field(default_factory=HQAddress)


class BookExtraction(BaseModel):
    title: Optional[str] = None
    author_full_name: Optional[str] = None
    work_type: Optional[str] = None
    publication_year: Optional[str] = None

    award: AwardInfo = Field(default_factory=AwardInfo)
    affiliation: AffiliationInfo = Field(default_factory=AffiliationInfo)
    publisher: PublisherInfo = Field(default_factory=PublisherInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_book() -> str:
    return (
        "Extract structured information for a single book described in the answer that meets the task criteria. "
        "If multiple books are mentioned, choose the first one that appears to meet the constraints and extract its data. "
        "Return JSON with the following fields:\n"
        "- title: the book's title (string)\n"
        "- author_full_name: the author's full name (string)\n"
        "- work_type: the work type as stated (e.g., 'novel', 'short story collection', 'poetry') (string)\n"
        "- publication_year: the book's publication year (string, not number)\n"
        "- award: object containing:\n"
        "    • award_name: name of the award (string)\n"
        "    • award_year: year of the award (string)\n"
        "    • award_category: category of the award (string)\n"
        "    • award_urls: array of URL strings explicitly provided in the answer that verify the award win (official award page preferred)\n"
        "- affiliation: object containing:\n"
        "    • university_name: name of the California university where the author currently teaches (string)\n"
        "    • university_state: state name or abbreviation (string; e.g., 'California' or 'CA')\n"
        "    • position_title: the author's current academic position title (string; e.g., 'Professor', 'Associate Professor')\n"
        "    • affiliation_urls: array of URL strings explicitly provided in the answer that verify the author's current professorial position at that university\n"
        "- publisher: object containing:\n"
        "    • publisher_name: name of the publisher (string)\n"
        "    • headquarters: object containing:\n"
        "        · street_address: complete street address (string); it must include 'Broadway' if present in the answer\n"
        "        · city: city name (string; e.g., 'New York', 'New York City', or 'NYC')\n"
        "        · zip_code: ZIP code (string)\n"
        "        · location_urls: array of URL strings explicitly provided in the answer that verify the publisher's headquarters address\n\n"
        "Rules:\n"
        "1) Extract only what is explicitly present in the answer; if a field is missing, return null (or empty array for URLs).\n"
        "2) For URLs, only include actual URLs present in the answer text; do not invent or infer any.\n"
        "3) Do not convert types; keep years and ZIP codes as strings.\n"
        "4) If the answer mentions multiple URLs for a verification, include all of them in the corresponding URLs array.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_str(s: Optional[str]) -> str:
    return (s or "").strip()


def _parse_year_int(year_str: Optional[str]) -> Optional[int]:
    if not year_str:
        return None
    m = re.search(r"\b(20\d{2})\b", year_str)
    try:
        if m:
            return int(m.group(1))
        # direct int conversion fallback
        return int(year_str.strip())
    except Exception:
        return None


def check_publication_year_in_range(year_str: Optional[str]) -> bool:
    y = _parse_year_int(year_str)
    return y is not None and 2022 <= y <= 2024


def check_award_name_allowed(name: Optional[str]) -> bool:
    if not name:
        return False
    n = name.lower()
    allowed_patterns = [
        "national book award",
        "pulitzer prize",
        "pen/faulkner award",
        "pen faulkner award",
    ]
    return any(pat in n for pat in allowed_patterns)


def check_award_year_valid(year_str: Optional[str]) -> bool:
    y = _parse_year_int(year_str)
    return y in {2023, 2024}


def check_award_category_fiction(cat: Optional[str]) -> bool:
    if not cat:
        return False
    return "fiction" in cat.lower()


def check_state_is_california(state_str: Optional[str]) -> bool:
    if not state_str:
        return False
    s = state_str.strip().lower()
    return s in {"california", "ca", "ca.", "calif."}


def check_professorial_title(title: Optional[str]) -> bool:
    if not title:
        return False
    t = title.lower()
    prof_tokens = [
        "professor",
        "assistant professor",
        "associate professor",
        "adjunct professor",
        "distinguished professor",
        "emeritus",
        "professor of practice",
        "clinical professor",
        "visiting professor",
        "research professor",
        "chair professor",
        "endowed professor",
    ]
    disallowed = [
        "lecturer",
        "instructor",
        "teacher",
        "writer-in-residence",
        "staff",
        "postdoc",
    ]
    if any(bad in t for bad in disallowed):
        return False
    return any(tok in t for tok in prof_tokens)


def check_city_is_nyc(city: Optional[str]) -> bool:
    if not city:
        return False
    c = city.strip().lower()
    return c in {"new york", "new york city", "nyc", "new york, ny", "new york, nyc"}


def check_street_contains_broadway(street: Optional[str]) -> bool:
    if not street:
        return False
    return "broadway" in street.lower()


def check_zip_valid(zip_code: Optional[str]) -> bool:
    if not zip_code:
        return False
    # Accept 5-digit ZIP (e.g., 10007), optionally ZIP+4 like 10007-1234
    return bool(re.match(r"^\d{5}(-\d{4})?$", zip_code.strip()))


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_award_information_checks(
    evaluator: Evaluator,
    parent_node,
    book: BookExtraction,
) -> None:
    award_node = evaluator.add_parallel(
        id="Award_Information",
        desc="Provide and verify the qualifying major U.S. literary fiction award details",
        parent=parent_node,
        critical=True,
    )

    # Award Name allowed check
    evaluator.add_custom_node(
        result=check_award_name_allowed(book.award.award_name),
        id="Award_Name",
        desc="Award must be one of: National Book Award, Pulitzer Prize, or PEN/Faulkner Award",
        parent=award_node,
        critical=True,
    )

    # Award Year is 2023 or 2024
    evaluator.add_custom_node(
        result=check_award_year_valid(book.award.award_year),
        id="Award_Year",
        desc="Award year must be 2023 or 2024",
        parent=award_node,
        critical=True,
    )

    # Award Category is Fiction
    evaluator.add_custom_node(
        result=check_award_category_fiction(book.award.award_category),
        id="Award_Category",
        desc="Award category must be Fiction",
        parent=award_node,
        critical=True,
    )

    # Award Reference URL verification
    award_ref_leaf = evaluator.add_leaf(
        id="Award_Reference_URL",
        desc="Provide a URL that verifies the award win (name/year/category as applicable)",
        parent=award_node,
        critical=True,
    )
    award_claim = (
        f"The book '{_normalize_str(book.title)}' by {_normalize_str(book.author_full_name)} "
        f"won the {_normalize_str(book.award.award_name)} in {_normalize_str(book.award.award_year)} "
        f"in the {_normalize_str(book.award.award_category)} category."
    )
    await evaluator.verify(
        claim=award_claim,
        node=award_ref_leaf,
        sources=book.award.award_urls if book.award.award_urls else None,
        additional_instruction=(
            "Verify on the provided award page(s) that this book and author won the specified award, "
            "in the stated year, and in the Fiction category. Minor variations in naming are acceptable "
            "(e.g., 'Pulitzer Prize for Fiction'). If the URL does not clearly support the claim, mark as not supported."
        ),
    )


async def build_affiliation_checks(
    evaluator: Evaluator,
    parent_node,
    book: BookExtraction,
) -> None:
    aff_node = evaluator.add_parallel(
        id="Author_University_Affiliation",
        desc="Provide and verify the author's current professorial university position in California",
        parent=parent_node,
        critical=True,
    )

    # University Name provided
    evaluator.add_custom_node(
        result=bool(_normalize_str(book.affiliation.university_name)),
        id="University_Name",
        desc="Provide the name of the California university where the author currently teaches",
        parent=aff_node,
        critical=True,
    )

    # University State must be California
    evaluator.add_custom_node(
        result=check_state_is_california(book.affiliation.university_state),
        id="University_State",
        desc="University state must be California",
        parent=aff_node,
        critical=True,
    )

    # Professorial Position Title must be professorial
    evaluator.add_custom_node(
        result=check_professorial_title(book.affiliation.position_title),
        id="Professorial_Position_Title",
        desc="Provide the author's current academic position title; it must be a professorial title (i.e., a professor rank/title, not merely any staff/lecturer role)",
        parent=aff_node,
        critical=True,
    )

    # Affiliation Reference URL verification
    aff_ref_leaf = evaluator.add_leaf(
        id="Affiliation_Reference_URL",
        desc="Provide a URL verifying the author's current professorial position at the stated California university",
        parent=aff_node,
        critical=True,
    )
    aff_claim = (
        f"{_normalize_str(book.author_full_name)} currently holds the professorial title "
        f"'{_normalize_str(book.affiliation.position_title)}' at {_normalize_str(book.affiliation.university_name)} "
        f"in California."
    )
    await evaluator.verify(
        claim=aff_claim,
        node=aff_ref_leaf,
        sources=book.affiliation.affiliation_urls if book.affiliation.affiliation_urls else None,
        additional_instruction=(
            "Verify that the author currently holds a professorial position (e.g., Professor, Associate Professor) "
            "at the specified California university on the provided page(s). Accept minor title variations. "
            "If the page suggests past employment or non-professorial roles (e.g., lecturer), mark as not supported."
        ),
    )


async def build_publisher_checks(
    evaluator: Evaluator,
    parent_node,
    book: BookExtraction,
) -> None:
    pub_node = evaluator.add_parallel(
        id="Publisher_Information",
        desc="Provide and verify the publisher and its Broadway NYC headquarters address",
        parent=parent_node,
        critical=True,
    )

    # Publisher Name verification: try to confirm publisher of the book via award URLs
    pub_name_leaf = evaluator.add_leaf(
        id="Publisher_Name",
        desc="Provide the publisher's name",
        parent=pub_node,
        critical=True,
    )
    pub_claim = (
        f"The publisher of '{_normalize_str(book.title)}' is '{_normalize_str(book.publisher.publisher_name)}'."
    )
    # Prefer award URLs to confirm publisher; if none, fallback to location URLs (may not confirm the book-publisher linkage)
    publisher_sources: Optional[List[str]] = None
    if book.award.award_urls:
        publisher_sources = book.award.award_urls
    elif book.publisher.headquarters.location_urls:
        publisher_sources = book.publisher.headquarters.location_urls

    await evaluator.verify(
        claim=pub_claim,
        node=pub_name_leaf,
        sources=publisher_sources,
        additional_instruction=(
            "Confirm on the provided page(s) that the named publisher published the specified book. "
            "If the provided URL does not establish the book-publisher relationship, mark as not supported."
        ),
    )

    # Publisher Headquarters Address checks
    addr_node = evaluator.add_parallel(
        id="Publisher_Headquarters_Address",
        desc="Provide and verify the publisher headquarters address; must be on Broadway in New York City and include street address and ZIP code",
        parent=pub_node,
        critical=True,
    )

    # Street Address must include Broadway
    evaluator.add_custom_node(
        result=check_street_contains_broadway(book.publisher.headquarters.street_address),
        id="Street_Address",
        desc="Provide the street address; must be on Broadway",
        parent=addr_node,
        critical=True,
    )

    # City must be New York City
    evaluator.add_custom_node(
        result=check_city_is_nyc(book.publisher.headquarters.city),
        id="City",
        desc="City must be New York City",
        parent=addr_node,
        critical=True,
    )

    # ZIP Code must be present and valid format
    evaluator.add_custom_node(
        result=check_zip_valid(book.publisher.headquarters.zip_code),
        id="ZIP_Code",
        desc="Provide the ZIP code for the headquarters address",
        parent=addr_node,
        critical=True,
    )

    # Location Reference URL verification
    loc_ref_leaf = evaluator.add_leaf(
        id="Location_Reference_URL",
        desc="Provide a URL verifying the publisher headquarters address on Broadway in New York City",
        parent=addr_node,
        critical=True,
    )
    addr_claim = (
        f"The headquarters address of '{_normalize_str(book.publisher.publisher_name)}' is "
        f"{_normalize_str(book.publisher.headquarters.street_address)}, "
        f"{_normalize_str(book.publisher.headquarters.city)}, NY {_normalize_str(book.publisher.headquarters.zip_code)}, "
        f"and the street is Broadway."
    )
    await evaluator.verify(
        claim=addr_claim,
        node=loc_ref_leaf,
        sources=book.publisher.headquarters.location_urls if book.publisher.headquarters.location_urls else None,
        additional_instruction=(
            "Verify on the provided page(s) that the publisher's headquarters address is located on Broadway in New York City "
            "and includes the street address and ZIP code. Minor formatting differences are acceptable."
        ),
    )


async def build_book_main_checks(
    evaluator: Evaluator,
    root,
    book: BookExtraction,
) -> None:
    book_node = evaluator.add_parallel(
        id="Book_Information",
        desc="Provide complete information about a single book that meets all specified criteria",
        parent=root,
        critical=True,
    )

    # Book Title provided
    evaluator.add_custom_node(
        result=bool(_normalize_str(book.title)),
        id="Book_Title",
        desc="Provide the book's title",
        parent=book_node,
        critical=True,
    )

    # Author Full Name provided
    evaluator.add_custom_node(
        result=bool(_normalize_str(book.author_full_name)),
        id="Author_Full_Name",
        desc="Provide the author's full name",
        parent=book_node,
        critical=True,
    )

    # Book Is Novel verification (use award URLs if available)
    is_novel_leaf = evaluator.add_leaf(
        id="Book_Is_Novel",
        desc="Verify the book is a novel (fiction work), consistent with the task requirement",
        parent=book_node,
        critical=True,
    )
    novel_claim = f"The book '{_normalize_str(book.title)}' is a novel (a long-form work of fiction)."
    await evaluator.verify(
        claim=novel_claim,
        node=is_novel_leaf,
        sources=book.award.award_urls if book.award.award_urls else None,
        additional_instruction=(
            "Determine from the provided page(s) whether the work is a novel (not a poetry collection or story collection). "
            "If the page indicates the work is a novel, mark supported; otherwise mark not supported. "
            "If no URLs are provided, rely on the answer context."
        ),
    )

    # Publication Year checks: existence and validity (range 2022–2024)
    evaluator.add_custom_node(
        result=bool(_normalize_str(book.publication_year)),
        id="Publication_Year_Provided",
        desc="Publication year is provided in the answer",
        parent=book_node,
        critical=True,
    )

    pub_year_leaf = evaluator.add_leaf(
        id="Publication_Year",
        desc="Provide the book's publication year; must be between 2022 and 2024 (inclusive)",
        parent=book_node,
        critical=True,
    )
    pub_year_claim = (
        f"The book '{_normalize_str(book.title)}' was published in {_normalize_str(book.publication_year)}, "
        f"which falls within 2022 to 2024 inclusive."
    )
    # Use award URLs if present to corroborate publication year; otherwise simple verify
    await evaluator.verify(
        claim=pub_year_claim,
        node=pub_year_leaf,
        sources=book.award.award_urls if book.award.award_urls else None,
        additional_instruction=(
            "Check whether the claimed publication year is explicitly supported or consistent on the provided page(s). "
            "If the year is outside 2022–2024, mark as not supported."
        ),
    )

    # Subsections
    await build_award_information_checks(evaluator, book_node, book)
    await build_affiliation_checks(evaluator, book_node, book)
    await build_publisher_checks(evaluator, book_node, book)


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

    # Extract all relevant book information from the answer
    book_info = await evaluator.extract(
        prompt=prompt_extract_book(),
        template_class=BookExtraction,
        extraction_name="book_extraction",
    )

    # Build verification tree and perform checks
    await build_book_main_checks(evaluator, root, book_info)

    # Return evaluation summary
    return evaluator.get_summary()