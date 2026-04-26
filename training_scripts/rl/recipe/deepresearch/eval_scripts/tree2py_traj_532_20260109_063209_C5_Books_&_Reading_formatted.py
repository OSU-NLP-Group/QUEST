import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "award_fiction_2024"
TASK_DESCRIPTION = (
    "I am compiling a reading list featuring recent award-winning fiction for my book club, focusing on major U.S. "
    "literary awards from 2024. Please identify four fiction books that won the following prestigious awards in 2024: "
    "(1) The National Book Award for Fiction, (2) The Pulitzer Prize for Fiction, (3) The PEN/Faulkner Award for Fiction, "
    "and (4) The Andrew Carnegie Medal for Excellence in Fiction. For each book, provide the book title, author name, "
    "publisher, and publication date (month and year). Please ensure all information is accurate and verifiable from "
    "official award sources or publisher records."
)

AWARD_LABELS = {
    "nba": "National Book Award for Fiction",
    "pulitzer": "Pulitzer Prize for Fiction",
    "penfaulkner": "PEN/Faulkner Award for Fiction",
    "carnegie": "Andrew Carnegie Medal for Excellence in Fiction",
}


class BookRecord(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None  # Month + Year (string)
    award_urls: List[str] = Field(default_factory=list)  # Official award announcement/source URLs
    publication_urls: List[str] = Field(default_factory=list)  # Publisher or official database URLs


class AwardsExtraction(BaseModel):
    nba: Optional[BookRecord] = None
    pulitzer: Optional[BookRecord] = None
    penfaulkner: Optional[BookRecord] = None
    carnegie: Optional[BookRecord] = None


def prompt_extract_award_books() -> str:
    return (
        "Extract the four fiction books mentioned in the answer that correspond to these 2024 awards: "
        "1) National Book Award for Fiction, 2) Pulitzer Prize for Fiction, 3) PEN/Faulkner Award for Fiction, "
        "4) Andrew Carnegie Medal for Excellence in Fiction. "
        "For each award category, extract the following fields exactly as stated in the answer:\n"
        "- title: complete book title\n"
        "- author: author's full name\n"
        "- publisher: publisher's name\n"
        "- publication_date: publication date including month and year (e.g., 'April 2024', '04/2024', 'April 15, 2024')\n"
        "- award_urls: all URLs cited that point to official award sources/announcements confirming the win (e.g., nationalbook.org, pulitzer.org, penfaulkner.org, ala.org)\n"
        "- publication_urls: all URLs cited that point to publisher sources or official book databases confirming publisher and/or publication date\n\n"
        "Organize the JSON under these keys: 'nba', 'pulitzer', 'penfaulkner', 'carnegie'. "
        "If the answer does not provide a field, set it to null (or empty list for URLs). "
        "Only extract URLs explicitly present in the answer text (including markdown links). Do not invent URLs."
    )


def has_month_and_year(date_str: Optional[str]) -> bool:
    if not date_str or not isinstance(date_str, str):
        return False
    s = date_str.strip().lower()
    if not s:
        return False
    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    year_match = re.search(r"\b(19|20)\d{2}\b", s)
    month_name_present = any(m in s for m in months)
    mm_yyyy = re.search(r"\b(0?[1-9]|1[0-2])[/-](19|20)\d{2}\b", s)  # e.g., 04/2024 or 4-2024
    yyyy_mm = re.search(r"\b(19|20)\d{2}[/-](0?[1-9]|1[0-2])\b", s)  # e.g., 2024-04 or 2024/4
    iso_date = re.search(r"\b(19|20)\d{2}-(0[1-9]|1[0-2])-\d{2}\b", s)  # e.g., 2024-04-15
    if not year_match:
        return False
    return bool(month_name_present or mm_yyyy or yyyy_mm or iso_date)


def combine_sources(record: BookRecord) -> List[str]:
    seen = set()
    combined: List[str] = []
    for url in (record.award_urls or []):
        u = (url or "").strip()
        if u and u not in seen:
            seen.add(u)
            combined.append(u)
    for url in (record.publication_urls or []):
        u = (url or "").strip()
        if u and u not in seen:
            seen.add(u)
            combined.append(u)
    return combined


async def verify_award_book(
    evaluator: Evaluator,
    root_parent_node,
    record: Optional[BookRecord],
    award_key: str,
    group_node_id: str,
    group_node_desc: str,
    leaf_prefix: str,
) -> None:
    group_node = evaluator.add_parallel(
        id=group_node_id,
        desc=group_node_desc,
        parent=root_parent_node,
        critical=False  # Allow partial credit per award group
    )

    # Gracefully handle missing record
    title_val = (record.title.strip() if (record and record.title) else "")
    author_val = (record.author.strip() if (record and record.author) else "")
    publisher_val = (record.publisher.strip() if (record and record.publisher) else "")
    pub_date_val = (record.publication_date.strip() if (record and record.publication_date) else "")
    award_urls = (record.award_urls if (record and record.award_urls) else [])
    publication_urls = (record.publication_urls if (record and record.publication_urls) else [])
    combined_urls = combine_sources(record or BookRecord())

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(title_val),
        id=f"{leaf_prefix}_Title_Provided",
        desc="Complete book title is provided.",
        parent=group_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(author_val),
        id=f"{leaf_prefix}_Author_Full_Name_Provided",
        desc="Author's full name is provided.",
        parent=group_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(publisher_val),
        id=f"{leaf_prefix}_Publisher_Provided",
        desc="Publisher name is provided.",
        parent=group_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_month_and_year(pub_date_val),
        id=f"{leaf_prefix}_Publication_Date_Month_Year_Provided",
        desc="Publication date is provided and includes month and year.",
        parent=group_node,
        critical=True
    )

    # Award source cited (critical) - ensure at least one award URL provided
    evaluator.add_custom_node(
        result=(len(award_urls) > 0),
        id=f"{leaf_prefix}_Award_Source_Cited",
        desc="Provides a verifiable citation/URL to an official award organization website or official award announcement confirming the win.",
        parent=group_node,
        critical=True
    )

    # Winner correct (critical) - validate via award URLs
    winner_node = evaluator.add_leaf(
        id=f"{leaf_prefix}_Award_Winner_Correct",
        desc=f"Selected book is the correct 2024 {AWARD_LABELS[award_key]} winner (award-year/category match).",
        parent=group_node,
        critical=True
    )
    winner_claim = (
        f"The book titled '{title_val}' by {author_val} is the 2024 {AWARD_LABELS[award_key]} winner."
        " If the award category explicitly says 'Fiction', ensure the page confirms the Fiction category."
    )
    await evaluator.verify(
        claim=winner_claim,
        node=winner_node,
        sources=award_urls,
        additional_instruction=(
            "Use the provided official award URLs to confirm the 2024 winner in the Fiction category for the specified award."
            " Allow minor variations in naming/capitalization."
        ),
    )

    # Is fiction work (critical) - verify via award and/or publication pages
    fiction_node = evaluator.add_leaf(
        id=f"{leaf_prefix}_Is_Fiction_Work",
        desc="Selected book is a fiction work.",
        parent=group_node,
        critical=True
    )
    fiction_claim = (
        f"The book '{title_val}' is a work of fiction (e.g., a novel or short story collection)."
    )
    await evaluator.verify(
        claim=fiction_claim,
        node=fiction_node,
        sources=combined_urls,
        additional_instruction=(
            "Check genre/category labels on the award announcement or publisher page. "
            "Treat labels such as 'Fiction', 'Novel', 'Short Stories' as valid confirmations the work is fiction."
        ),
    )

    # Publication source cited (critical) - verify that publisher and/or pub date is supported by publication URLs
    pubsrc_node = evaluator.add_leaf(
        id=f"{leaf_prefix}_Publication_Source_Cited",
        desc="Provides a verifiable citation/URL to a publisher source or official book database confirming publication information (publisher and/or publication date).",
        parent=group_node,
        critical=True
    )
    # Build a flexible claim: at least one of the publication facts should be supported
    pieces = []
    if publisher_val:
        pieces.append(f"the publisher is '{publisher_val}'")
    if pub_date_val:
        pieces.append(f"the publication date includes '{pub_date_val}'")
    if pieces:
        pubsrc_claim = (
            f"At least one of the following facts is confirmed on the page for '{title_val}': "
            + " or ".join(pieces)
            + "."
        )
    else:
        pubsrc_claim = (
            f"This page is a credible publisher or official catalog entry for '{title_val}'."
        )

    await evaluator.verify(
        claim=pubsrc_claim,
        node=pubsrc_node,
        sources=publication_urls,
        additional_instruction=(
            "Confirm at least one publication detail (publisher or month+year publication date) is present on the page. "
            "If both are present, that also satisfies the requirement."
        ),
    )


async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_award_books(),
        template_class=AwardsExtraction,
        extraction_name="awards_extraction",
    )

    evaluator.add_ground_truth({
        "required_awards": list(AWARD_LABELS.values()),
        "year": 2024,
        "info_required_per_book": ["title", "author", "publisher", "publication_date (month+year)"],
        "source_requirements": [
            "Official award announcement URL(s)",
            "Publisher or official database URL(s)"
        ]
    })

    await verify_award_book(
        evaluator=evaluator,
        root_parent_node=root,
        record=extracted.nba or BookRecord(),
        award_key="nba",
        group_node_id="Book_1_National_Book_Award_Fiction_2024",
        group_node_desc="Book that won the National Book Award for Fiction in 2024, with required bibliographic fields and verification.",
        leaf_prefix="NBA"
    )

    await verify_award_book(
        evaluator=evaluator,
        root_parent_node=root,
        record=extracted.pulitzer or BookRecord(),
        award_key="pulitzer",
        group_node_id="Book_2_Pulitzer_Prize_Fiction_2024",
        group_node_desc="Book that won the Pulitzer Prize for Fiction in 2024, with required bibliographic fields and verification.",
        leaf_prefix="Pulitzer"
    )

    await verify_award_book(
        evaluator=evaluator,
        root_parent_node=root,
        record=extracted.penfaulkner or BookRecord(),
        award_key="penfaulkner",
        group_node_id="Book_3_PEN_Faulkner_Award_Fiction_2024",
        group_node_desc="Book that won the PEN/Faulkner Award for Fiction in 2024, with required bibliographic fields and verification.",
        leaf_prefix="PENFaulkner"
    )

    await verify_award_book(
        evaluator=evaluator,
        root_parent_node=root,
        record=extracted.carnegie or BookRecord(),
        award_key="carnegie",
        group_node_id="Book_4_Andrew_Carnegie_Medal_Fiction_2024",
        group_node_desc="Book that won the Andrew Carnegie Medal for Excellence in Fiction in 2024, with required bibliographic fields and verification.",
        leaf_prefix="Carnegie"
    )

    return evaluator.get_summary()