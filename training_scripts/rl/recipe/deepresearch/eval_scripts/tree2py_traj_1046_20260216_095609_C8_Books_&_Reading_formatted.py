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
TASK_ID = "literary_awards_industry_report_2024"
TASK_DESCRIPTION = """
Compile a comprehensive report on the 2024 literary awards landscape and book industry trends in the United States and United Kingdom. The report must include winners and details for major awards, multiple-award identification, Booker page-count and shortest ranking information, 2024 U.S. audiobook statistics, and 2024 ALA censorship data, with reference URLs for each major category.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AwardEntry(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class BookerEntry(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    page_count: Optional[str] = None  # Keep as string to allow formats like "192 pages"
    shortest_ranking: Optional[str] = None  # e.g., "second-shortest", "No. 2", etc.
    reference_urls: List[str] = Field(default_factory=list)


class FirstNovelEntry(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    organization: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class MultipleAwardEntry(BaseModel):
    book_title: Optional[str] = None
    author: Optional[str] = None
    claim_text: Optional[str] = None  # The exact statement used by the answer (if provided)
    reference_urls: List[str] = Field(default_factory=list)


class AudiobookStats(BaseModel):
    revenue: Optional[str] = None            # e.g., "$2.22 billion"
    growth_rate: Optional[str] = None        # e.g., "13%"
    digital_share: Optional[str] = None      # e.g., "99%"
    reference_urls: List[str] = Field(default_factory=list)


class CensorshipStats(BaseModel):
    most_challenged_title: Optional[str] = None
    most_challenged_author: Optional[str] = None
    attempts_count: Optional[str] = None     # e.g., "821"
    unique_titles_count: Optional[str] = None  # e.g., "2,452"
    reference_urls: List[str] = Field(default_factory=list)


class ReportExtraction(BaseModel):
    nba: AwardEntry = AwardEntry()
    pulitzer: AwardEntry = AwardEntry()
    booker: BookerEntry = BookerEntry()
    first_novel: FirstNovelEntry = FirstNovelEntry()
    multiple_awards: MultipleAwardEntry = MultipleAwardEntry()
    audiobook_stats: AudiobookStats = AudiobookStats()
    censorship_stats: CensorshipStats = CensorshipStats()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_report() -> str:
    return """
    Extract structured information from the answer for the following categories. Use exactly the fields specified and extract URLs explicitly present in the answer text. If any field is not mentioned, set it to null (for strings) or an empty list (for URLs).

    1) National Book Award for Fiction (2024) -> nba:
       - title: The complete title of the winning book
       - author: The full author name
       - publisher: The publisher or imprint (if the answer provides it)
       - reference_urls: All URLs the answer cites that confirm the NBA winner details

    2) Pulitzer Prize for Fiction (2024) -> pulitzer:
       - title
       - author
       - publisher
       - reference_urls: All URLs cited that confirm the Pulitzer winner details

    3) Booker Prize (2024) -> booker:
       - title
       - author
       - publisher
       - page_count: Total page count of the winning book, as stated
       - shortest_ranking: A short phrase summarizing its historical ranking among the shortest Booker winners, as stated (e.g., "second-shortest")
       - reference_urls: All URLs cited that confirm the Booker winner, page count, and shortest ranking information

    4) Center for Fiction's First Novel Prize (2024) -> first_novel:
       - title
       - author
       - organization: The awarding organization's name (e.g., The Center for Fiction) as presented by the answer
       - reference_urls: All URLs cited that confirm the First Novel Prize winner details

    5) Multiple Award Winner Identification (2024-2025) -> multiple_awards:
       - book_title: The book claimed to have won multiple major awards
       - author: The author of that book
       - claim_text: The exact multi-award claim if present (e.g., "James by Percival Everett won both the 2024 National Book Award and 2025 Pulitzer Prize for Fiction")
       - reference_urls: All URLs cited that support this multi-award claim

    6) Audiobook Industry Statistics (U.S., 2024) -> audiobook_stats:
       - revenue: Total audiobook sales revenue (string as presented, e.g., "$2.22 billion")
       - growth_rate: Percentage growth rate compared to 2023 (string, e.g., "13%")
       - digital_share: Market share percentage of digital audiobooks (string, e.g., "99%")
       - reference_urls: All URLs cited that support these audiobook statistics

    7) Book Censorship Statistics (ALA, 2024) -> censorship_stats:
       - most_challenged_title: The title of the #1 most challenged book
       - most_challenged_author: The author of the #1 most challenged book
       - attempts_count: Total number of censorship attempts documented
       - unique_titles_count: Number of unique book titles targeted for censorship
       - reference_urls: All URLs cited that support these censorship statistics
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _sources_exist(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls or [])


def _safe_str(value: Optional[str]) -> str:
    return value or ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_national_book_award(evaluator: Evaluator, root, data: AwardEntry) -> None:
    parent = evaluator.add_parallel(
        id="National_Book_Award_Fiction_2024",
        desc="Information about the 2024 National Book Award for Fiction winner",
        parent=root,
        critical=False
    )

    # Existence of references
    evaluator.add_custom_node(
        result=_sources_exist(data.reference_urls),
        id="NBA_Reference_Exists",
        desc="National Book Award: Reference URLs are provided",
        parent=parent,
        critical=True
    )

    # Reference support
    nba_ref_leaf = evaluator.add_leaf(
        id="NBA_Reference_URL",
        desc="Valid reference URL confirming the National Book Award winner",
        parent=parent,
        critical=True
    )
    ref_claim = f"The provided sources confirm that the 2024 National Book Award for Fiction was awarded to '{_safe_str(data.title)}' by '{_safe_str(data.author)}'."
    await evaluator.verify(
        claim=ref_claim,
        node=nba_ref_leaf,
        sources=data.reference_urls,
        additional_instruction="Check the sources for the official winner announcement. Allow minor variations in punctuation/casing."
    )

    # Title
    nba_title_leaf = evaluator.add_leaf(
        id="NBA_Book_Title",
        desc="Correct title of the winning book",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The winning book title for the 2024 National Book Award for Fiction is '{_safe_str(data.title)}'.",
        node=nba_title_leaf,
        sources=data.reference_urls,
        additional_instruction="Verify the stated book title against the official National Book Award announcement page or credible news coverage."
    )

    # Author
    nba_author_leaf = evaluator.add_leaf(
        id="NBA_Author",
        desc="Correct author name",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The author of '{_safe_str(data.title)}' (NBA 2024 Fiction winner) is '{_safe_str(data.author)}'.",
        node=nba_author_leaf,
        sources=data.reference_urls,
        additional_instruction="Allow minor variants (middle initials, casing). Confirm author identity from the sources."
    )

    # Publisher
    nba_publisher_leaf = evaluator.add_leaf(
        id="NBA_Publisher",
        desc="Correct publisher name",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publisher of '{_safe_str(data.title)}' is '{_safe_str(data.publisher)}'.",
        node=nba_publisher_leaf,
        sources=data.reference_urls,
        additional_instruction="Confirm the publisher/imprint from credible sources. Allow imprint-parent equivalence (e.g., Little, Brown vs Hachette)."
    )


async def verify_pulitzer(evaluator: Evaluator, root, data: AwardEntry) -> None:
    parent = evaluator.add_parallel(
        id="Pulitzer_Prize_Fiction_2024",
        desc="Information about the 2024 Pulitzer Prize for Fiction winner",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=_sources_exist(data.reference_urls),
        id="Pulitzer_Reference_Exists",
        desc="Pulitzer Prize: Reference URLs are provided",
        parent=parent,
        critical=True
    )

    pul_ref = evaluator.add_leaf(
        id="Pulitzer_Reference_URL",
        desc="Valid reference URL confirming the Pulitzer Prize winner",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided sources confirm that the 2024 Pulitzer Prize for Fiction was awarded to '{_safe_str(data.title)}' by '{_safe_str(data.author)}'.",
        node=pul_ref,
        sources=data.reference_urls,
        additional_instruction="Verify via the official Pulitzer site or authoritative coverage."
    )

    pul_title = evaluator.add_leaf(
        id="Pulitzer_Book_Title",
        desc="Correct title of the winning book",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Pulitzer Prize for Fiction (2024) winning book title is '{_safe_str(data.title)}'.",
        node=pul_title,
        sources=data.reference_urls,
        additional_instruction="Confirm the exact title from the Pulitzer announcement page or reliable coverage."
    )

    pul_author = evaluator.add_leaf(
        id="Pulitzer_Author",
        desc="Correct author name",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The author of the 2024 Pulitzer Fiction winner '{_safe_str(data.title)}' is '{_safe_str(data.author)}'.",
        node=pul_author,
        sources=data.reference_urls,
        additional_instruction="Allow minor variants; confirm identity."
    )

    pul_publisher = evaluator.add_leaf(
        id="Pulitzer_Publisher",
        desc="Correct publisher name",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publisher of '{_safe_str(data.title)}' is '{_safe_str(data.publisher)}'.",
        node=pul_publisher,
        sources=data.reference_urls,
        additional_instruction="If multiple imprints/editions exist, ensure the cited publisher aligns with the sources."
    )


async def verify_booker(evaluator: Evaluator, root, data: BookerEntry) -> None:
    parent = evaluator.add_parallel(
        id="Booker_Prize_2024",
        desc="Information about the 2024 Booker Prize winner including page count and historical ranking",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=_sources_exist(data.reference_urls),
        id="Booker_Reference_Exists",
        desc="Booker Prize: Reference URLs are provided",
        parent=parent,
        critical=True
    )

    ref_leaf = evaluator.add_leaf(
        id="Booker_Reference_URL",
        desc="Valid reference URL confirming the Booker Prize winner and page count",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided sources confirm that the 2024 Booker Prize was awarded to '{_safe_str(data.title)}' by '{_safe_str(data.author)}', and provide page-count information.",
        node=ref_leaf,
        sources=data.reference_urls,
        additional_instruction="Use the official Booker site or credible coverage that includes both winner and page-count info."
    )

    title_leaf = evaluator.add_leaf(
        id="Booker_Book_Title",
        desc="Correct title of the winning book",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2024 Booker Prize winning book title is '{_safe_str(data.title)}'.",
        node=title_leaf,
        sources=data.reference_urls,
        additional_instruction="Confirm from official Booker page."
    )

    author_leaf = evaluator.add_leaf(
        id="Booker_Author",
        desc="Correct author name",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The author of the 2024 Booker winner '{_safe_str(data.title)}' is '{_safe_str(data.author)}'.",
        node=author_leaf,
        sources=data.reference_urls,
        additional_instruction="Allow minor variants; confirm author identity."
    )

    page_leaf = evaluator.add_leaf(
        id="Booker_Page_Count",
        desc="Correct page count of the winning book",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total page count of '{_safe_str(data.title)}' is '{_safe_str(data.page_count)}'.",
        node=page_leaf,
        sources=data.reference_urls,
        additional_instruction="Extract/confirm the numeric page count; allow formatting like '192 pages'."
    )

    ranking_leaf = evaluator.add_leaf(
        id="Booker_Shortest_Ranking",
        desc="Correct ranking as second-shortest Booker Prize winner in history",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2024 Booker Prize winner '{_safe_str(data.title)}' is the second-shortest Booker Prize winner in history.",
        node=ranking_leaf,
        sources=data.reference_urls,
        additional_instruction="Confirm from credible analysis or official Booker materials; allow mention of ties."
    )


async def verify_first_novel_prize(evaluator: Evaluator, root, data: FirstNovelEntry) -> None:
    parent = evaluator.add_parallel(
        id="First_Novel_Prize_2024",
        desc="Information about the 2024 Center for Fiction First Novel Prize winner",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=_sources_exist(data.reference_urls),
        id="FirstNovel_Reference_Exists",
        desc="First Novel Prize: Reference URLs are provided",
        parent=parent,
        critical=True
    )

    ref_leaf = evaluator.add_leaf(
        id="FirstNovel_Reference_URL",
        desc="Valid reference URL confirming the First Novel Prize winner",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The sources confirm the 2024 Center for Fiction First Novel Prize winner: '{_safe_str(data.title)}' by '{_safe_str(data.author)}'.",
        node=ref_leaf,
        sources=data.reference_urls,
        additional_instruction="Use official Center for Fiction site or credible coverage."
    )

    title_leaf = evaluator.add_leaf(
        id="FirstNovel_Book_Title",
        desc="Correct title of the winning book",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2024 First Novel Prize winning title is '{_safe_str(data.title)}'.",
        node=title_leaf,
        sources=data.reference_urls,
        additional_instruction="Confirm from the Center for Fiction announcement."
    )

    author_leaf = evaluator.add_leaf(
        id="FirstNovel_Author",
        desc="Correct author name",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The author of the First Novel Prize winner '{_safe_str(data.title)}' is '{_safe_str(data.author)}'.",
        node=author_leaf,
        sources=data.reference_urls,
        additional_instruction="Allow minor variants; confirm identity."
    )

    org_leaf = evaluator.add_leaf(
        id="FirstNovel_Organization",
        desc="Correct awarding organization name",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The awarding organization for the First Novel Prize is '{_safe_str(data.organization)}'.",
        node=org_leaf,
        sources=data.reference_urls,
        additional_instruction="Confirm that the prize is given by The Center for Fiction."
    )


async def verify_multiple_award(evaluator: Evaluator, root, data: MultipleAwardEntry) -> None:
    parent = evaluator.add_parallel(
        id="Multiple_Award_Winner_Identification",
        desc="Identification of any book that won multiple major awards in 2024-2025",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=_sources_exist(data.reference_urls),
        id="Multiple_Award_Ref_Exists",
        desc="Multiple-award identification: Reference URLs are provided",
        parent=parent,
        critical=True
    )

    ref_leaf = evaluator.add_leaf(
        id="Multiple_Award_Reference_URL",
        desc="Valid reference URL confirming the multiple awards",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources confirm a claim that a single book won multiple major awards across the 2024-2025 season.",
        node=ref_leaf,
        sources=data.reference_urls,
        additional_instruction="Focus on official award pages or credible reporting."
    )

    # Specific rubric claim: 'James' by Percival Everett won NBA 2024 and Pulitzer 2025 for Fiction
    multi_leaf = evaluator.add_leaf(
        id="Multiple_Award_Book",
        desc="Correct identification that 'James' by Percival Everett won both the 2024 National Book Award and 2025 Pulitzer Prize for Fiction",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The book 'James' by Percival Everett won the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction.",
        node=multi_leaf,
        sources=data.reference_urls,
        additional_instruction="Confirm both awards for the same book 'James' by Percival Everett; allow minor date phrasing."
    )


async def verify_audiobook_stats(evaluator: Evaluator, root, data: AudiobookStats) -> None:
    parent = evaluator.add_parallel(
        id="Audiobook_Industry_Statistics_2024",
        desc="Statistics about the 2024 audiobook market",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=_sources_exist(data.reference_urls),
        id="Audiobook_Ref_Exists",
        desc="Audiobook statistics: Reference URLs are provided",
        parent=parent,
        critical=True
    )

    ref_leaf = evaluator.add_leaf(
        id="Audiobook_Reference_URL",
        desc="Valid reference URL confirming audiobook statistics",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources substantiate the 2024 U.S. audiobook revenue, the YoY growth rate vs 2023, and the digital audiobook market share.",
        node=ref_leaf,
        sources=data.reference_urls,
        additional_instruction="Use credible industry reports; allow minor formatting differences (commas, currency signs)."
    )

    revenue_leaf = evaluator.add_leaf(
        id="Audiobook_Revenue",
        desc="Correct 2024 audiobook sales revenue figure ($2.22 billion)",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total U.S. audiobook sales revenue for 2024 was '{_safe_str(data.revenue)}'.",
        node=revenue_leaf,
        sources=data.reference_urls,
        additional_instruction="Confirm the revenue figure from the source; allow formats like $2.22B or $2.22 billion."
    )

    growth_leaf = evaluator.add_leaf(
        id="Audiobook_Growth_Rate",
        desc="Correct growth percentage from 2023 to 2024 (13%)",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The percentage growth rate for U.S. audiobook sales from 2023 to 2024 was '{_safe_str(data.growth_rate)}'.",
        node=growth_leaf,
        sources=data.reference_urls,
        additional_instruction="Confirm the stated YoY percentage from the cited report."
    )

    share_leaf = evaluator.add_leaf(
        id="Digital_Market_Share",
        desc="Correct digital audiobook market share percentage (99%)",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The market share percentage of digital audiobooks in 2024 was '{_safe_str(data.digital_share)}'.",
        node=share_leaf,
        sources=data.reference_urls,
        additional_instruction="Confirm the digital share (downloads/streams) per the cited source; allow 'approximate' if clearly stated."
    )


async def verify_censorship_stats(evaluator: Evaluator, root, data: CensorshipStats) -> None:
    parent = evaluator.add_parallel(
        id="Book_Censorship_Statistics_2024",
        desc="Statistics about book censorship and challenges in 2024 tracked by the American Library Association",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=_sources_exist(data.reference_urls),
        id="Censorship_Ref_Exists",
        desc="ALA censorship: Reference URLs are provided",
        parent=parent,
        critical=True
    )

    ref_leaf = evaluator.add_leaf(
        id="Censorship_Reference_URL",
        desc="Valid reference URL confirming censorship statistics",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The sources substantiate the ALA-tracked 2024 censorship statistics (most-challenged book, attempts total, and unique titles targeted).",
        node=ref_leaf,
        sources=data.reference_urls,
        additional_instruction="Use ALA's official reporting or credible summaries."
    )

    title_leaf = evaluator.add_leaf(
        id="Most_Challenged_Book_Title",
        desc="Correct title of the #1 most challenged book",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The #1 most challenged book title in 2024 per ALA was '{_safe_str(data.most_challenged_title)}'.",
        node=title_leaf,
        sources=data.reference_urls,
        additional_instruction="Confirm from ALA report; allow minor punctuation/casing differences."
    )

    author_leaf = evaluator.add_leaf(
        id="Most_Challenged_Book_Author",
        desc="Correct author of the #1 most challenged book",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The author of the #1 most challenged book '{_safe_str(data.most_challenged_title)}' was '{_safe_str(data.most_challenged_author)}'.",
        node=author_leaf,
        sources=data.reference_urls,
        additional_instruction="Confirm the author from ALA report."
    )

    attempts_leaf = evaluator.add_leaf(
        id="Censorship_Attempts_Count",
        desc="Correct number of censorship attempts tracked by ALA (821)",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total number of censorship attempts documented by ALA in 2024 was '{_safe_str(data.attempts_count)}'.",
        node=attempts_leaf,
        sources=data.reference_urls,
        additional_instruction="Numbers may include thousands separators; focus on numeric equivalence."
    )

    unique_leaf = evaluator.add_leaf(
        id="Unique_Titles_Targeted",
        desc="Correct number of unique titles targeted (2,452)",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The number of unique book titles targeted for censorship in 2024 was '{_safe_str(data.unique_titles_count)}'.",
        node=unique_leaf,
        sources=data.reference_urls,
        additional_instruction="Confirm numeric count; allow thousands separators."
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
    Evaluate an answer for the 2024 literary awards and industry report task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel as per rubric
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

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_report(),
        template_class=ReportExtraction,
        extraction_name="literary_awards_industry_report_extraction",
    )

    # Add top-level report node (optional wrapper to mirror rubric naming)
    report_node = evaluator.add_parallel(
        id="2024_Literary_Awards_and_Industry_Report",
        desc="Complete report covering 2024 major literary award winners and book industry statistics",
        parent=root,
        critical=False
    )

    # Build and run verifications for each rubric category
    await verify_national_book_award(evaluator, report_node, extracted.nba)
    await verify_pulitzer(evaluator, report_node, extracted.pulitzer)
    await verify_booker(evaluator, report_node, extracted.booker)
    await verify_first_novel_prize(evaluator, report_node, extracted.first_novel)
    await verify_multiple_award(evaluator, report_node, extracted.multiple_awards)
    await verify_audiobook_stats(evaluator, report_node, extracted.audiobook_stats)
    await verify_censorship_stats(evaluator, report_node, extracted.censorship_stats)

    # Return structured summary
    return evaluator.get_summary()