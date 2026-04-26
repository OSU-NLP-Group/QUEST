import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lit_prize_2025_single_book"
TASK_DESCRIPTION = (
    "In 2025, three major English-language literary fiction prizes were awarded: "
    "the Pulitzer Prize for Fiction, the Booker Prize, and the National Book Award for Fiction. "
    "Identify one book that won one of these three prizes in 2025 and meets all of the following criteria:\n\n"
    "1. The book was published between January 1, 2024, and December 31, 2025 (inclusive), and its publication date preceded the award announcement.\n"
    "2. The author had previously won or been shortlisted for at least one major literary award before winning their 2025 prize.\n"
    "3. The book is at least the author's third published work of fiction (not a debut or second novel).\n"
    "4. The publisher of the book has previously published at least one winner in the same award category (i.e., if the book won the Booker Prize, the publisher must have published a previous Booker Prize winner; if it won the Pulitzer Prize for Fiction, the publisher must have published a previous Pulitzer Prize for Fiction winner; and so on).\n"
    "5. The author has lived or worked in at least two different countries during their career.\n\n"
    "Provide the book's title, author's full name, the specific prize it won, the publisher, and the publication date. Include URL references that verify each of these criteria."
)

ALLOWED_PRIZES = [
    "Pulitzer Prize for Fiction",
    "Booker Prize",
    "National Book Award for Fiction",
]
DATE_RANGE_START = "2024-01-01"
DATE_RANGE_END = "2025-12-31"
AWARD_YEAR = "2025"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BookExtraction(BaseModel):
    """
    Unified extraction structure for the identified book and all evidences.
    All URLs should be explicitly present in the answer; if any are missing, return empty lists.
    """
    # Core details
    title: Optional[str] = None
    author_full_name: Optional[str] = None
    publisher_name: Optional[str] = None
    prize_name: Optional[str] = None  # expected to be one of ALLOWED_PRIZES
    publication_date: Optional[str] = None  # Accept any reasonable date format as string
    language: Optional[str] = None  # e.g., 'English' or 'English translation'

    # Award timing
    award_year: Optional[str] = None  # e.g., '2025'
    award_announcement_date: Optional[str] = None  # date string if provided

    # Evidence URLs
    award_urls: List[str] = Field(default_factory=list)  # Official award announcement URLs
    publication_urls: List[str] = Field(default_factory=list)  # Publisher or reliable source pages confirming pub date
    language_urls: List[str] = Field(default_factory=list)  # Pages confirming English language (original or translation)
    author_awards_urls: List[str] = Field(default_factory=list)  # Author's prior recognition evidences
    bibliography_urls: List[str] = Field(default_factory=list)  # Author bibliography/work count pages
    publisher_history_urls: List[str] = Field(default_factory=list)  # Publisher's previous wins in same category
    biography_urls: List[str] = Field(default_factory=list)  # Author biography pages confirming geographic background

    # Optional contextual details (strings for flexibility)
    prior_awards: List[str] = Field(default_factory=list)  # Names/descriptions of prior award wins or shortlist nominations
    fiction_work_count: Optional[str] = None  # e.g., '3', 'three novels', '4+'
    countries: List[str] = Field(default_factory=list)  # countries lived/worked


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_book() -> str:
    return """
    Extract the single book identified in the answer that won a major 2025 English-language literary fiction prize
    (Pulitzer Prize for Fiction, Booker Prize, or National Book Award for Fiction). Extract all required fields and
    the specific URL evidences mentioned in the answer that verify each criterion.

    REQUIRED FIELDS:
    - title: The book title.
    - author_full_name: The author's full name.
    - publisher_name: The publisher name.
    - prize_name: The specific prize the book won (e.g., 'Pulitzer Prize for Fiction', 'Booker Prize', 'National Book Award for Fiction').
    - publication_date: The book's publication date (string; any reasonable format is acceptable).
    - language: The language of publication (e.g., 'English' or 'English translation').

    AWARD TIMING:
    - award_year: The year the prize was awarded (string, expected '2025' if stated).
    - award_announcement_date: The award announcement date (string) if the answer provides it.

    EVIDENCE URLS (extract only URLs that are explicitly present in the answer; if missing, return empty lists):
    - award_urls: Official award announcement page(s) confirming the book as the 2025 winner.
    - publication_urls: Publisher or reliable database pages confirming the publication date.
    - language_urls: Pages confirming that the book is published in English (either original language or translation).
    - author_awards_urls: Pages confirming the author's prior award wins or shortlist nominations before 2025.
    - bibliography_urls: Pages listing the author's fiction works or otherwise confirming work count (to show this book is at least the third).
    - publisher_history_urls: Pages confirming that the publisher has published at least one previous winner in the same award category.
    - biography_urls: Pages confirming the author's geographic background, indicating living or working in at least two countries.

    OPTIONAL CONTEXT (strings for flexibility; if absent, use null or empty):
    - prior_awards: A list of names/descriptions of the author's prior award wins or shortlist nominations.
    - fiction_work_count: A string indicating the count of the author's fiction works (e.g., '3', 'three novels', '4+').
    - countries: A list of countries the author has lived or worked in.

    IMPORTANT RULES:
    1. Extract ONLY what appears in the answer. Do not invent or infer information or URLs.
    2. For any field not provided in the answer, return null (for single fields) or an empty list (for URLs or multi-value lists).
    3. For URLs, support plain URLs and markdown links; extract the actual URL string.
    4. Do not include duplicate URLs; return each unique URL once.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _combine_sources(*lists: List[str]) -> List[str]:
    """Combine multiple URL lists while preserving order and removing duplicates/empties."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for url in lst or []:
            u = (url or "").strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_basic_book_information(
    evaluator: Evaluator, parent_node, info: BookExtraction
) -> None:
    """Basic info presence checks (critical parallel)."""
    basic_node = evaluator.add_parallel(
        id="Basic_Book_Information",
        desc="Verification that all required book details are provided in the answer",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(info.title),
        id="Title_Provided",
        desc="The book's title is provided",
        parent=basic_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(info.author_full_name),
        id="Author_Name_Provided",
        desc="The author's full name is provided",
        parent=basic_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(info.publisher_name),
        id="Publisher_Provided",
        desc="The publisher's name is provided",
        parent=basic_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(info.prize_name),
        id="Prize_Identified",
        desc="The specific prize won (Pulitzer Prize for Fiction, Booker Prize, or National Book Award for Fiction) is identified",
        parent=basic_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(info.publication_date),
        id="Publication_Date_Provided",
        desc="The publication date is provided",
        parent=basic_node,
        critical=True,
    )


async def build_award_eligibility(
    evaluator: Evaluator, parent_node, info: BookExtraction
) -> None:
    """Award eligibility subtree (critical sequential)."""
    award_root = evaluator.add_sequential(
        id="Award_Eligibility",
        desc="Verification of award winner status and language eligibility",
        parent=parent_node,
        critical=True,
    )

    # Award Winner Verification (parallel critical)
    winner_node = evaluator.add_parallel(
        id="Award_Winner_Verification",
        desc="Verification that the book won one of the three specified prizes in 2025",
        parent=award_root,
        critical=True,
    )

    # Is Major Prize
    is_major_leaf = evaluator.add_leaf(
        id="Is_Major_Prize",
        desc="The prize is the Pulitzer Prize for Fiction, Booker Prize, or National Book Award for Fiction",
        parent=winner_node,
        critical=True,
    )
    claim_major = (
        f"The prize '{info.prize_name or ''}' is one of the following major prizes: "
        f"Pulitzer Prize for Fiction, Booker Prize, or National Book Award for Fiction. "
        f"Accept reasonable naming variants such as 'The Booker Prize' or 'Pulitzer Prize (Fiction)'."
    )
    await evaluator.verify(
        claim=claim_major,
        node=is_major_leaf,
        sources=info.award_urls,
        additional_instruction=(
            "Use the award announcement URL(s) to confirm the prize naming, and accept well-known naming variants. "
            "If no URLs are provided, conclude this check as not supported."
        ),
    )

    # Won in 2025
    won_2025_leaf = evaluator.add_leaf(
        id="Won_In_2025",
        desc="The prize was awarded in the year 2025",
        parent=winner_node,
        critical=True,
    )
    claim_won_2025 = (
        f"This book won the '{info.prize_name or ''}' in 2025."
    )
    await evaluator.verify(
        claim=claim_won_2025,
        node=won_2025_leaf,
        sources=info.award_urls,
        additional_instruction=(
            "Verify the award year on the official announcement page(s). "
            "If no URLs are provided, conclude this check as not supported."
        ),
    )

    # English language publication (leaf critical)
    english_leaf = evaluator.add_leaf(
        id="English_Language_Publication",
        desc="The book is published in English (original or translation)",
        parent=award_root,
        critical=True,
    )
    lang_sources = _combine_sources(info.language_urls, info.publication_urls, info.award_urls)
    claim_english = (
        "The book is published in English, either originally written in English or available as an English translation."
    )
    await evaluator.verify(
        claim=claim_english,
        node=english_leaf,
        sources=lang_sources,
        additional_instruction=(
            "Confirm English-language status using the provided URLs (publisher pages, official announcements, or other reliable sources). "
            "If no URLs are provided, conclude this check as not supported."
        ),
    )

    # Award Evidence (parallel critical)
    award_evidence_node = evaluator.add_parallel(
        id="Award_Evidence",
        desc="Documentation verifying the award",
        parent=award_root,
        critical=True,
    )
    award_verify_leaf = evaluator.add_leaf(
        id="Award_Verification_URL",
        desc="URL reference from official award announcement confirming the book as 2025 winner",
        parent=award_evidence_node,
        critical=True,
    )
    claim_award_verify = (
        f"The official award announcement page confirms that the book '{info.title or ''}' "
        f"won the {AWARD_YEAR} {info.prize_name or ''}."
    )
    await evaluator.verify(
        claim=claim_award_verify,
        node=award_verify_leaf,
        sources=info.award_urls,
        additional_instruction=(
            "Use the official award announcement URL(s) to verify the winner status. "
            "If no URLs are provided, conclude this check as not supported."
        ),
    )


async def build_publication_requirements(
    evaluator: Evaluator, parent_node, info: BookExtraction
) -> None:
    """Publication requirements subtree (critical sequential)."""
    pub_root = evaluator.add_sequential(
        id="Publication_Requirements",
        desc="Verification of publication date and timing relative to award announcement",
        parent=parent_node,
        critical=True,
    )

    # Publication Timing (parallel critical)
    timing_node = evaluator.add_parallel(
        id="Publication_Timing",
        desc="Verification that publication occurred in the eligible timeframe and before award",
        parent=pub_root,
        critical=True,
    )

    # Published within 2024–2025 inclusive
    published_leaf = evaluator.add_leaf(
        id="Published_2024_Or_2025",
        desc="The publication date is between January 1, 2024 and December 31, 2025 (inclusive)",
        parent=timing_node,
        critical=True,
    )
    claim_published_range = (
        f"The book's publication date ({info.publication_date or 'unknown'}) is between {DATE_RANGE_START} and {DATE_RANGE_END}, inclusive."
    )
    await evaluator.verify(
        claim=claim_published_range,
        node=published_leaf,
        sources=info.publication_urls,
        additional_instruction=(
            "Confirm the publication date on the publisher or reliable source pages provided. "
            "If no URLs are provided, conclude this check as not supported."
        ),
    )

    # Publication precedes award announcement date
    before_award_leaf = evaluator.add_leaf(
        id="Before_Award_Date",
        desc="The publication date precedes the award announcement date",
        parent=timing_node,
        critical=True,
    )
    timing_sources = _combine_sources(info.publication_urls, info.award_urls)
    claim_before_award = (
        f"The book's publication date ({info.publication_date or 'unknown'}) is earlier than the award announcement date "
        f"({info.award_announcement_date or 'unknown'})."
    )
    await evaluator.verify(
        claim=claim_before_award,
        node=before_award_leaf,
        sources=timing_sources,
        additional_instruction=(
            "Verify the relative ordering of the publication date and the award announcement date using the provided URLs. "
            "If either date is missing or no URLs are provided, conclude this check as not supported."
        ),
    )

    # Publication Evidence (parallel critical)
    pub_evidence_node = evaluator.add_parallel(
        id="Publication_Evidence",
        desc="Documentation verifying the publication date",
        parent=pub_root,
        critical=True,
    )
    pub_verify_leaf = evaluator.add_leaf(
        id="Publication_Verification_URL",
        desc="URL reference from publisher or reliable source confirming the publication date",
        parent=pub_evidence_node,
        critical=True,
    )
    claim_pub_verify = (
        f"The provided publication source confirms the book's publication date as {info.publication_date or 'unknown'}."
    )
    await evaluator.verify(
        claim=claim_pub_verify,
        node=pub_verify_leaf,
        sources=info.publication_urls,
        additional_instruction=(
            "Use publisher or reliable database pages (e.g., catalog, bibliographic entries) to confirm the publication date. "
            "If no URLs are provided, conclude this check as not supported."
        ),
    )


async def build_author_credentials(
    evaluator: Evaluator, parent_node, info: BookExtraction
) -> None:
    """Author credentials subtree (critical parallel)."""
    author_root = evaluator.add_parallel(
        id="Author_Credentials",
        desc="Verification of author's prior recognition and career stage",
        parent=parent_node,
        critical=True,
    )

    # Prior Recognition (sequential critical)
    prior_root = evaluator.add_sequential(
        id="Prior_Recognition",
        desc="Verification that the author had previous major literary award recognition before the 2025 prize",
        parent=author_root,
        critical=True,
    )

    prev_status_node = evaluator.add_parallel(
        id="Previous_Award_Status",
        desc="Verification of the author's previous award history",
        parent=prior_root,
        critical=True,
    )
    has_prior_leaf = evaluator.add_leaf(
        id="Has_Prior_Recognition",
        desc="The author has at least one previous major literary award win or shortlist nomination",
        parent=prev_status_node,
        critical=True,
    )
    claim_has_prior = (
        "Before the 2025 prize, the author had at least one major literary award win or shortlist nomination "
        "(shortlist counts; longlist alone does not suffice)."
    )
    await evaluator.verify(
        claim=claim_has_prior,
        node=has_prior_leaf,
        sources=info.author_awards_urls,
        additional_instruction=(
            "Confirm prior major recognition (wins or shortlist nominations) from the provided URLs. "
            "Examples include Booker, Pulitzer, National Book Award, Women's Prize, etc. "
            "If no URLs are provided, conclude this check as not supported."
        ),
    )

    prior_before_leaf = evaluator.add_leaf(
        id="Recognition_Before_2025",
        desc="The previous award recognition occurred before the 2025 prize was awarded",
        parent=prev_status_node,
        critical=True,
    )
    prior_vs_award_sources = _combine_sources(info.author_awards_urls, info.award_urls)
    claim_prior_before = (
        "The author's previous award recognition (win or shortlist) occurred before the 2025 prize was awarded."
    )
    await evaluator.verify(
        claim=claim_prior_before,
        node=prior_before_leaf,
        sources=prior_vs_award_sources,
        additional_instruction=(
            "Use the dates on the award history pages and the 2025 award announcement page(s) to confirm ordering. "
            "If dates or URLs are missing, conclude this check as not supported."
        ),
    )

    award_hist_evidence = evaluator.add_parallel(
        id="Award_History_Evidence",
        desc="Documentation of author's award history",
        parent=prior_root,
        critical=True,
    )
    award_hist_leaf = evaluator.add_leaf(
        id="Award_History_URL",
        desc="URL reference confirming the author's previous award wins or nominations",
        parent=award_hist_evidence,
        critical=True,
    )
    claim_award_hist = (
        "The provided URL(s) confirm the author's previous award wins or shortlist nominations."
    )
    await evaluator.verify(
        claim=claim_award_hist,
        node=award_hist_leaf,
        sources=info.author_awards_urls,
        additional_instruction=(
            "If no URLs are provided, conclude this check as not supported."
        ),
    )

    # Career Maturity (sequential critical)
    career_root = evaluator.add_sequential(
        id="Career_Maturity",
        desc="Verification that the book is not a debut or second novel",
        parent=author_root,
        critical=True,
    )
    work_count_node = evaluator.add_parallel(
        id="Work_Count",
        desc="Verification of the author's publication history",
        parent=career_root,
        critical=True,
    )
    third_or_later_leaf = evaluator.add_leaf(
        id="Third_Or_Later_Fiction",
        desc="The book is at least the author's third published work of fiction",
        parent=work_count_node,
        critical=True,
    )
    claim_third_or_later = (
        "This book is at least the author's third published work of fiction (not a debut or second). "
        "Count novels or short story collections as fiction; exclude poetry and non-fiction."
    )
    await evaluator.verify(
        claim=claim_third_or_later,
        node=third_or_later_leaf,
        sources=info.bibliography_urls,
        additional_instruction=(
            "Use the bibliography source(s) to count fiction works. "
            "If no URLs are provided, conclude this check as not supported."
        ),
    )

    biblio_evidence_node = evaluator.add_parallel(
        id="Bibliography_Evidence",
        desc="Documentation of author's bibliography",
        parent=career_root,
        critical=True,
    )
    biblio_leaf = evaluator.add_leaf(
        id="Bibliography_URL",
        desc="URL reference confirming the author's complete fiction bibliography or work count",
        parent=biblio_evidence_node,
        critical=True,
    )
    claim_biblio = (
        "The provided bibliography source(s) confirm the author's fiction works such that this book is at least the third."
    )
    await evaluator.verify(
        claim=claim_biblio,
        node=biblio_leaf,
        sources=info.bibliography_urls,
        additional_instruction=(
            "If no URLs are provided, conclude this check as not supported."
        ),
    )


async def build_publisher_credentials(
    evaluator: Evaluator, parent_node, info: BookExtraction
) -> None:
    """Publisher credentials subtree (critical sequential)."""
    pubcred_root = evaluator.add_sequential(
        id="Publisher_Credentials",
        desc="Verification that the publisher has previously won the same award category",
        parent=parent_node,
        critical=True,
    )

    pub_hist_node = evaluator.add_parallel(
        id="Publisher_Award_History",
        desc="Verification of publisher's history with the specific award",
        parent=pubcred_root,
        critical=True,
    )
    has_prev_winner_leaf = evaluator.add_leaf(
        id="Has_Previous_Winner",
        desc="The publisher has published at least one previous winner in the same award category that the book won in 2025",
        parent=pub_hist_node,
        critical=True,
    )
    claim_prev_winner = (
        f"The publisher '{info.publisher_name or ''}' has previously published at least one winner of the same award category "
        f"that this book won ({info.prize_name or ''}). Accept imprints under the same publishing group as valid."
    )
    await evaluator.verify(
        claim=claim_prev_winner,
        node=has_prev_winner_leaf,
        sources=info.publisher_history_urls,
        additional_instruction=(
            "Use the provided publisher history URL(s) to confirm prior winners in the same category. "
            "Imprints under the same corporate publisher count. "
            "If no URLs are provided, conclude this check as not supported."
        ),
    )

    pub_hist_evidence = evaluator.add_parallel(
        id="Publisher_History_Evidence",
        desc="Documentation of publisher's award history",
        parent=pubcred_root,
        critical=True,
    )
    pub_hist_leaf = evaluator.add_leaf(
        id="Publisher_History_URL",
        desc="URL reference confirming the publisher's previous wins in the same award category",
        parent=pub_hist_evidence,
        critical=True,
    )
    claim_pub_hist = (
        "The provided URL(s) confirm that the publisher previously published at least one winner in the same award category."
    )
    await evaluator.verify(
        claim=claim_pub_hist,
        node=pub_hist_leaf,
        sources=info.publisher_history_urls,
        additional_instruction=(
            "If no URLs are provided, conclude this check as not supported."
        ),
    )


async def build_author_geographic_background(
    evaluator: Evaluator, parent_node, info: BookExtraction
) -> None:
    """Author geographic background subtree (critical sequential)."""
    geo_root = evaluator.add_sequential(
        id="Author_Geographic_Background",
        desc="Verification of the author's international background and experience",
        parent=parent_node,
        critical=True,
    )

    geo_exp_node = evaluator.add_parallel(
        id="Geographic_Experience",
        desc="Verification of multi-country residence or work experience",
        parent=geo_root,
        critical=True,
    )
    multi_country_leaf = evaluator.add_leaf(
        id="Multi_Country_Background",
        desc="The author has lived or worked in at least two different countries during their career",
        parent=geo_exp_node,
        critical=True,
    )
    claim_multi_country = (
        "The author has lived or worked in at least two different countries during their career."
    )
    await evaluator.verify(
        claim=claim_multi_country,
        node=multi_country_leaf,
        sources=info.biography_urls,
        additional_instruction=(
            "Use biography pages or interviews to confirm residence/work in multiple countries. "
            "If no URLs are provided, conclude this check as not supported."
        ),
    )

    bio_evidence_node = evaluator.add_parallel(
        id="Biography_Evidence",
        desc="Documentation of author's geographic background",
        parent=geo_root,
        critical=True,
    )
    bio_leaf = evaluator.add_leaf(
        id="Biography_URL",
        desc="URL reference confirming the author's geographic background and residence history",
        parent=bio_evidence_node,
        critical=True,
    )
    claim_bio_verify = (
        "The biography source(s) confirm the author's geographic background, including living or working in at least two countries."
    )
    await evaluator.verify(
        claim=claim_bio_verify,
        node=bio_leaf,
        sources=info.biography_urls,
        additional_instruction=(
            "If no URLs are provided, conclude this check as not supported."
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
    Evaluate an answer for the 2025 literary prize single-book identification task.
    """
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

    # Top-level task node (parallel, non-critical)
    task_node = evaluator.add_parallel(
        id="Book_Identification_Task",
        desc="Identify a book that won a major 2025 English-language literary fiction prize and meets all specified author, publisher, and publication criteria",
        parent=root,
        critical=False,
    )

    # Extract structured info from the answer
    info: BookExtraction = await evaluator.extract(
        prompt=prompt_extract_book(),
        template_class=BookExtraction,
        extraction_name="book_extraction",
    )

    # Add custom info to summary for transparency
    evaluator.add_custom_info(
        info={
            "allowed_prizes": ALLOWED_PRIZES,
            "award_year": AWARD_YEAR,
            "publication_date_range": [DATE_RANGE_START, DATE_RANGE_END],
        },
        info_type="evaluation_parameters",
        info_name="parameters",
    )

    # Build verification subtrees according to rubric
    await build_basic_book_information(evaluator, task_node, info)
    await build_award_eligibility(evaluator, task_node, info)
    await build_publication_requirements(evaluator, task_node, info)
    await build_author_credentials(evaluator, task_node, info)
    await build_publisher_credentials(evaluator, task_node, info)
    await build_author_geographic_background(evaluator, task_node, info)

    # Return structured summary
    return evaluator.get_summary()