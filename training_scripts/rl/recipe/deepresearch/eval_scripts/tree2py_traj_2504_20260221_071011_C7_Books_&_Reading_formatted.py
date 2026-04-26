import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "four_award_winning_books_2024"
TASK_DESCRIPTION = """
Identify four books published in 2024, each winning a different major literary award with the following specifications:

1. The National Book Award for Fiction winner that was published by Doubleday and is a retelling or reimagining of a classic American literary work.

2. The Pulitzer Prize for Fiction winner that was published by Knopf and is set during or after the American Civil War.

3. The Booker Prize winner that is set in space or aboard the International Space Station and was written by a British author.

4. The Goodreads Choice Award for Romance winner that was written by Emily Henry and published in April 2024.

For each book, provide the title and author.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BookInfo(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    award_name: Optional[str] = None
    award_category: Optional[str] = None
    award_year: Optional[str] = None
    award_result: Optional[str] = None  # e.g., "Winner"
    setting: Optional[str] = None       # e.g., "set in space", "post-Civil War"
    author_nationality: Optional[str] = None
    publication_date: Optional[str] = None   # e.g., "April 23, 2024"
    publication_month: Optional[str] = None  # e.g., "April"
    publication_year: Optional[str] = None   # e.g., "2024"
    special_note: Optional[str] = None       # e.g., retelling info
    sources: List[str] = Field(default_factory=list)


class FourBooksExtraction(BaseModel):
    nba_book: Optional[BookInfo] = None                 # National Book Award for Fiction (Doubleday + retelling)
    pulitzer_book: Optional[BookInfo] = None            # Pulitzer Prize for Fiction (Knopf + Civil War setting)
    booker_book: Optional[BookInfo] = None              # Booker Prize (space/ISS + British author)
    goodreads_romance_book: Optional[BookInfo] = None   # Goodreads Choice Award for Romance (Emily Henry + April 2024)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_four_books() -> str:
    return """
You will extract structured information for four specific 2024 award-winning books from the provided answer. Your job is to match the books in the answer to the four requirements below and extract the requested fields.

The four requirements to match are:
1) National Book Award for Fiction winner that was published by Doubleday and is a retelling or reimagining of a classic American literary work.
2) Pulitzer Prize for Fiction winner that was published by Knopf and is set during or after the American Civil War.
3) Booker Prize winner that is set in space or aboard the International Space Station and was written by a British author.
4) Goodreads Choice Award for Romance winner that was written by Emily Henry and was published in April 2024.

Instructions:
- Identify which book in the answer corresponds to each requirement and fill the fields accordingly.
- Extract only what is explicitly present in the answer text. Do not invent any information.
- Titles and authors must be exactly as written in the answer.
- For sources, extract all URLs explicitly mentioned in the answer that are relevant to that specific book (award announcements, publisher pages, author pages, reviews, Goodreads pages, etc.). If none are present for a book, return an empty list.
- If a field is not mentioned in the answer, return null for that field.

Return a JSON object with the following top-level fields, each an object with the fields below or null if the corresponding book wasn't provided:
- nba_book
- pulitzer_book
- booker_book
- goodreads_romance_book

For each book object, include these fields (use null if missing):
- title
- author
- publisher
- award_name
- award_category
- award_year
- award_result
- setting
- author_nationality
- publication_date
- publication_month
- publication_year
- special_note
- sources (array of URLs)
""".strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_book_label(info: Optional[BookInfo]) -> str:
    if not info:
        return "the book"
    t = info.title or ""
    a = info.author or ""
    if t and a:
        return f"'{t}' by {a}"
    if t:
        return f"'{t}'"
    if a:
        return f"the book by {a}"
    return "the book"


def _sources_or_empty(info: Optional[BookInfo]) -> List[str]:
    return (info.sources if (info and info.sources) else [])


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_book_1_verification(evaluator: Evaluator, parent_node, info: Optional[BookInfo]) -> None:
    """
    Book 1: National Book Award for Fiction winner published by Doubleday; retelling/reimagining of a classic American work.
    """
    book_node = evaluator.add_parallel(
        id="Book_1_National_Book_Award",
        desc="Book 1: The National Book Award for Fiction winner published by Doubleday that is a retelling of a classic American work",
        parent=parent_node,
        critical=False
    )

    label = _safe_book_label(info)
    srcs = _sources_or_empty(info)

    # Leaf 1: Won National Book Award for Fiction in 2024
    nba_node = evaluator.add_leaf(
        id="Book_1_NBA_Won",
        desc="Book 1 won the National Book Award for Fiction in 2024",
        parent=book_node,
        critical=True
    )
    claim_nba = f"The book {label} won the National Book Award for Fiction in 2024."
    await evaluator.verify(
        claim=claim_nba,
        node=nba_node,
        sources=srcs,
        additional_instruction="Verify that the provided pages explicitly indicate this book is the 2024 National Book Award (Fiction) winner. If a list of winners is given, confirm this specific title is listed under Fiction and that it is the winner."
    )

    # Leaf 2: Published by Doubleday
    pub_node = evaluator.add_leaf(
        id="Book_1_Publisher",
        desc="Book 1 was published by Doubleday",
        parent=book_node,
        critical=True
    )
    claim_pub = f"The book {label} was published by Doubleday."
    await evaluator.verify(
        claim=claim_pub,
        node=pub_node,
        sources=srcs,
        additional_instruction="Look for the publisher field on publisher pages, book retailer pages, or official announcements indicating Doubleday as the publisher (imprint of Knopf Doubleday Publishing Group is acceptable if explicitly labeled as Doubleday)."
    )

    # Leaf 3: Retelling/Reimagining of a classic American literary work
    retell_node = evaluator.add_leaf
    retell_node = evaluator.add_leaf(
        id="Book_1_Retelling",
        desc="Book 1 is a retelling or reimagining of a classic American literary work",
        parent=book_node,
        critical=True
    )
    claim_retell = f"The book {label} is a retelling or reimagining of a classic American literary work."
    await evaluator.verify(
        claim=claim_retell,
        node=retell_node,
        sources=srcs,
        additional_instruction="Check descriptions, reviews, or publisher copy that explicitly describe the book as a retelling or reimagining of a classic American literary work (e.g., referencing the original classic)."
    )


async def build_book_2_verification(evaluator: Evaluator, parent_node, info: Optional[BookInfo]) -> None:
    """
    Book 2: Pulitzer Prize for Fiction winner published by Knopf; set during or after the American Civil War.
    """
    book_node = evaluator.add_parallel(
        id="Book_2_Pulitzer_Prize",
        desc="Book 2: The Pulitzer Prize for Fiction winner published by Knopf set during or after the Civil War",
        parent=parent_node,
        critical=False
    )

    label = _safe_book_label(info)
    srcs = _sources_or_empty(info)

    # Leaf 1: Won Pulitzer Prize for Fiction in 2024
    pul_node = evaluator.add_leaf(
        id="Book_2_Pulitzer_Won",
        desc="Book 2 won the Pulitzer Prize for Fiction in 2024",
        parent=book_node,
        critical=True
    )
    claim_pul = f"The book {label} won the Pulitzer Prize for Fiction in 2024."
    await evaluator.verify(
        claim=claim_pul,
        node=pul_node,
        sources=srcs,
        additional_instruction="Verify that the provided sources explicitly indicate this book is the 2024 Pulitzer Prize for Fiction winner (not finalist)."
    )

    # Leaf 2: Published by Knopf
    pub_node = evaluator.add_leaf(
        id="Book_2_Publisher",
        desc="Book 2 was published by Knopf",
        parent=book_node,
        critical=True
    )
    claim_pub = f"The book {label} was published by Knopf."
    await evaluator.verify(
        claim=claim_pub,
        node=pub_node,
        sources=srcs,
        additional_instruction="Confirm the publisher is Alfred A. Knopf (often styled as 'Knopf') on official pages or trusted sources."
    )

    # Leaf 3: Setting during or after the American Civil War
    setting_node = evaluator.add_leaf(
        id="Book_2_Setting",
        desc="Book 2 is set during or after the American Civil War",
        parent=book_node,
        critical=True
    )
    claim_setting = f"The book {label} is set during the American Civil War or in its aftermath (post-1865)."
    await evaluator.verify(
        claim=claim_setting,
        node=setting_node,
        sources=srcs,
        additional_instruction="Look for plot summaries or descriptions explicitly placing the setting during the Civil War (1861–1865) or in the immediate post-war/Reconstruction period."
    )


async def build_book_3_verification(evaluator: Evaluator, parent_node, info: Optional[BookInfo]) -> None:
    """
    Book 3: Booker Prize winner set in space/ISS; written by a British author.
    """
    book_node = evaluator.add_parallel(
        id="Book_3_Booker_Prize",
        desc="Book 3: The Booker Prize winner set in space, written by a British author",
        parent=parent_node,
        critical=False
    )

    label = _safe_book_label(info)
    srcs = _sources_or_empty(info)

    # Leaf 1: Won Booker Prize in 2024
    booker_node = evaluator.add_leaf(
        id="Book_3_Booker_Won",
        desc="Book 3 won the Booker Prize in 2024",
        parent=book_node,
        critical=True
    )
    claim_booker = f"The book {label} won the Booker Prize in 2024."
    await evaluator.verify(
        claim=claim_booker,
        node=booker_node,
        sources=srcs,
        additional_instruction="Confirm that the sources clearly state this title is the 2024 Booker Prize winner (not longlisted or shortlisted only)."
    )

    # Leaf 2: Set in space or aboard the International Space Station
    setting_node = evaluator.add_leaf(
        id="Book_3_Setting",
        desc="Book 3 is set in space or aboard the International Space Station",
        parent=book_node,
        critical=True
    )
    claim_setting = f"The book {label} is set in space or aboard the International Space Station."
    await evaluator.verify(
        claim=claim_setting,
        node=setting_node,
        sources=srcs,
        additional_instruction="Look for explicit mentions that the story is set in outer space or on the ISS within synopses or reviews."
    )

    # Leaf 3: Written by a British author
    nationality_node = evaluator.add_leaf(
        id="Book_3_Author_Nationality",
        desc="Book 3 was written by a British author",
        parent=book_node,
        critical=True
    )
    claim_nat = f"The author of {label} is a British author."
    await evaluator.verify(
        claim=claim_nat,
        node=nationality_node,
        sources=srcs,
        additional_instruction="Verify author nationality from reliable sources (publisher bios, reputable profiles). Consider British as pertaining to the United Kingdom (England, Scotland, Wales, Northern Ireland)."
    )


async def build_book_4_verification(evaluator: Evaluator, parent_node, info: Optional[BookInfo]) -> None:
    """
    Book 4: Goodreads Choice Award for Romance winner by Emily Henry; published in April 2024.
    """
    book_node = evaluator.add_parallel(
        id="Book_4_Goodreads_Choice",
        desc="Book 4: The Goodreads Choice Award for Romance winner by Emily Henry published in April 2024",
        parent=parent_node,
        critical=False
    )

    label = _safe_book_label(info)
    srcs = _sources_or_empty(info)

    # Leaf 1: Won Goodreads Choice Award for Romance in 2024
    gr_node = evaluator.add_leaf(
        id="Book_4_Goodreads_Won",
        desc="Book 4 won the Goodreads Choice Award for Romance in 2024",
        parent=book_node,
        critical=True
    )
    claim_gr = f"The book {label} won the Goodreads Choice Award for Romance in 2024."
    await evaluator.verify(
        claim=claim_gr,
        node=gr_node,
        sources=srcs,
        additional_instruction="Confirm the Goodreads Choice Awards page (or reputable coverage) shows this title as the 2024 Romance winner."
    )

    # Leaf 2: Written by Emily Henry
    author_node = evaluator.add_leaf(
        id="Book_4_Author",
        desc="Book 4 was written by Emily Henry",
        parent=book_node,
        critical=True
    )
    claim_author = f"The author of {label} is Emily Henry."
    await evaluator.verify(
        claim=claim_author,
        node=author_node,
        sources=srcs,
        additional_instruction="Verify the book's author name on official book pages or trusted sources matches 'Emily Henry'."
    )

    # Leaf 3: Published in April 2024
    pubdate_node = evaluator.add_leaf(
        id="Book_4_Publication_Date",
        desc="Book 4 was published in April 2024",
        parent=book_node,
        critical=True
    )
    claim_pubdate = f"The book {label} was published in April 2024."
    await evaluator.verify(
        claim=claim_pubdate,
        node=pubdate_node,
        sources=srcs,
        additional_instruction="Accept any day within April 2024 (e.g., 'April 23, 2024'). The source must explicitly show an April 2024 publication date."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 'Four Award Winning Books 2024' task.
    """
    # Initialize evaluator (root is non-critical by design)
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

    # Extract structured data
    extraction = await evaluator.extract(
        prompt=prompt_extract_four_books(),
        template_class=FourBooksExtraction,
        extraction_name="extracted_books"
    )

    # Optional: add a top-level rubric node mirroring the provided JSON root (set non-critical to avoid strict constraint)
    rubric_root = evaluator.add_parallel(
        id="Four_Award_Winning_Books_2024",
        desc="Find four books published in 2024, each winning a different major literary award with specific attributes",
        parent=root,
        critical=False
    )

    # Build verification for each of the four books in parallel
    await asyncio.gather(
        build_book_1_verification(evaluator, rubric_root, extraction.nba_book),
        build_book_2_verification(evaluator, rubric_root, extraction.pulitzer_book),
        build_book_3_verification(evaluator, rubric_root, extraction.booker_book),
        build_book_4_verification(evaluator, rubric_root, extraction.goodreads_romance_book),
    )

    return evaluator.get_summary()