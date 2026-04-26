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
TASK_ID = "2024_lit_awards_bestsellers"
TASK_DESCRIPTION = """
Provide comprehensive factual information about four significant books from 2024 American and international literary landscape:

1. For the 2024 National Book Award for Fiction winner, provide: the book title, the author's name, and the publisher.

2. For the 2024 Pulitzer Prize for Fiction winner, provide: the book title, the author's name, the publisher, and the page count.

3. For the 2024 Booker Prize winner, provide: the book title, the author's name, the publisher, the page count, and any notable distinction related to the book's length in Booker Prize history.

4. For the book that spent the most weeks at #1 on the New York Times bestseller list in 2024, provide: the book title, the author's name, the publisher, and the specific number of weeks it held the #1 position.

Each piece of information must be supported by reference URLs from authoritative sources.
""".strip()

# Ground truth (expected official facts reflected in the rubric)
GROUND_TRUTH = {
    "National_Book_Award_Fiction_Winner": {
        "title": "James",
        "author": "Percival Everett",
        "publisher": "Doubleday",
    },
    "Pulitzer_Prize_Fiction_Winner": {
        "title": "Night Watch",
        "author": "Jayne Anne Phillips",
        "publisher": "Knopf (Alfred A. Knopf)",
        "page_count": "276",
    },
    "Booker_Prize_Winner": {
        "title": "Orbital",
        "author": "Samantha Harvey",
        "publisher": "Grove Press (or Grove Atlantic)",
        "page_count": "136",
        "length_distinction": "second-shortest novel to win the Booker Prize",
    },
    "NYT_Bestseller_Most_Weeks_Number_One": {
        "title": "The Women",
        "author": "Kristin Hannah",
        "publisher": "St. Martin's Press",
        "weeks_at_number_one": "10",
    }
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PrizeGroup(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    page_count: Optional[str] = None
    length_distinction: Optional[str] = None
    weeks_at_number_one: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LiteraryExtraction(BaseModel):
    nba_fiction: Optional[PrizeGroup] = None
    pulitzer_fiction: Optional[PrizeGroup] = None
    booker_prize: Optional[PrizeGroup] = None
    nyt_most_weeks_2024: Optional[PrizeGroup] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the following structured information from the answer text. Only extract what is explicitly stated in the answer. For each section, also extract all authoritative reference URLs mentioned that support the facts in that section.

Return a JSON object with the following top-level fields: nba_fiction, pulitzer_fiction, booker_prize, nyt_most_weeks_2024. Each field is an object with:
- title: string or null
- author: string or null
- publisher: string or null
- page_count: string or null (only if present/relevant)
- length_distinction: string or null (only if present/relevant)
- weeks_at_number_one: string or null (only if present/relevant)
- sources: array of URLs (if any URLs are mentioned in the answer for that section; otherwise empty array)

Section-specific guidance:
1) nba_fiction (2024 National Book Award for Fiction winner): title, author, publisher, and sources.
2) pulitzer_fiction (2024 Pulitzer Prize for Fiction winner): title, author, publisher, page_count, and sources.
3) booker_prize (2024 Booker Prize winner): title, author, publisher, page_count, length_distinction (about its length within Booker Prize history), and sources.
4) nyt_most_weeks_2024 (book that spent the most weeks at #1 on the New York Times bestseller list in 2024): title, author, publisher, weeks_at_number_one (a specific number), and sources.

Rules for URLs:
- Extract only valid URLs explicitly present in the answer.
- Include full URLs (prepend http:// if the protocol is missing).
- If multiple URLs are cited for one section, include all of them in the sources array of that section.
""".strip()


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(info: Optional[PrizeGroup]) -> List[str]:
    if info and info.sources:
        return info.sources
    return []


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_nba_section(evaluator: Evaluator, parent, info: Optional[PrizeGroup]) -> None:
    section_node = evaluator.add_parallel(
        id="National_Book_Award_Fiction_Winner",
        desc="Information about the 2024 National Book Award for Fiction winner",
        parent=parent,
        critical=True
    )

    sources_list = _safe_sources(info)
    evaluator.add_custom_node(
        result=len(sources_list) > 0,
        id="NBA_Reference_URL",
        desc="Authoritative reference URL(s) are provided to verify the National Book Award winner information",
        parent=section_node,
        critical=True
    )

    # Title
    nba_title_leaf = evaluator.add_leaf(
        id="NBA_Title",
        desc="The book title is 'James'",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="For the 2024 National Book Award for Fiction, the winning book is titled 'James'.",
        node=nba_title_leaf,
        sources=sources_list,
        additional_instruction="Allow case-insensitive matching and minor punctuation differences."
    )

    # Author
    nba_author_leaf = evaluator.add_leaf(
        id="NBA_Author",
        desc="The author is Percival Everett",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="For the 2024 National Book Award for Fiction, the winning book's author is Percival Everett.",
        node=nba_author_leaf,
        sources=sources_list,
        additional_instruction="Allow case-insensitive matching and minor variants (middle initials absent/present)."
    )

    # Publisher
    nba_publisher_leaf = evaluator.add_leaf(
        id="NBA_Publisher",
        desc="The publisher is Doubleday",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="The publisher of the 2024 National Book Award for Fiction winning book is Doubleday.",
        node=nba_publisher_leaf,
        sources=sources_list,
        additional_instruction="Accept 'Doubleday' as the imprint; minor label variations are acceptable if clearly the same imprint."
    )


async def verify_pulitzer_section(evaluator: Evaluator, parent, info: Optional[PrizeGroup]) -> None:
    section_node = evaluator.add_parallel(
        id="Pulitzer_Prize_Fiction_Winner",
        desc="Information about the 2024 Pulitzer Prize for Fiction winner",
        parent=parent,
        critical=True
    )

    sources_list = _safe_sources(info)
    evaluator.add_custom_node(
        result=len(sources_list) > 0,
        id="Pulitzer_Reference_URL",
        desc="Authoritative reference URL(s) are provided to verify the Pulitzer Prize winner information",
        parent=section_node,
        critical=True
    )

    # Title
    pul_title_leaf = evaluator.add_leaf(
        id="Pulitzer_Title",
        desc="The book title is 'Night Watch'",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="For the 2024 Pulitzer Prize for Fiction, the winning book is titled 'Night Watch'.",
        node=pul_title_leaf,
        sources=sources_list,
        additional_instruction="Allow case-insensitive matching and minor punctuation differences."
    )

    # Author
    pul_author_leaf = evaluator.add_leaf(
        id="Pulitzer_Author",
        desc="The author is Jayne Anne Phillips",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="For the 2024 Pulitzer Prize for Fiction, the winning book's author is Jayne Anne Phillips.",
        node=pul_author_leaf,
        sources=sources_list,
        additional_instruction="Allow case-insensitive matching and minor variants (middle initials absent/present)."
    )

    # Publisher
    pul_publisher_leaf = evaluator.add_leaf(
        id="Pulitzer_Publisher",
        desc="The publisher is Knopf (Alfred A. Knopf)",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="The publisher of 'Night Watch' (2024 Pulitzer Prize for Fiction winner) is Alfred A. Knopf (often referred to as Knopf).",
        node=pul_publisher_leaf,
        sources=sources_list,
        additional_instruction="Accept 'Knopf' or 'Alfred A. Knopf' as equivalent names for the same imprint."
    )

    # Page Count
    pul_pages_leaf = evaluator.add_leaf(
        id="Pulitzer_Page_Count",
        desc="The page count is 276 pages",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2024 Pulitzer Prize for Fiction winner 'Night Watch' has 276 pages (standard edition).",
        node=pul_pages_leaf,
        sources=sources_list,
        additional_instruction="Verify the stated page count on authoritative sources (e.g., publisher page, library catalogs). Accept '276' or '276 pages'."
    )


async def verify_booker_section(evaluator: Evaluator, parent, info: Optional[PrizeGroup]) -> None:
    section_node = evaluator.add_parallel(
        id="Booker_Prize_Winner",
        desc="Information about the 2024 Booker Prize winner",
        parent=parent,
        critical=True
    )

    sources_list = _safe_sources(info)
    evaluator.add_custom_node(
        result=len(sources_list) > 0,
        id="Booker_Reference_URL",
        desc="Authoritative reference URL(s) are provided to verify the Booker Prize winner information",
        parent=section_node,
        critical=True
    )

    # Title
    booker_title_leaf = evaluator.add_leaf(
        id="Booker_Title",
        desc="The book title is 'Orbital'",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="For the 2024 Booker Prize, the winning book is titled 'Orbital'.",
        node=booker_title_leaf,
        sources=sources_list,
        additional_instruction="Allow case-insensitive matching and minor punctuation differences."
    )

    # Author
    booker_author_leaf = evaluator.add_leaf(
        id="Booker_Author",
        desc="The author is Samantha Harvey",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="The author of the 2024 Booker Prize winning book is Samantha Harvey.",
        node=booker_author_leaf,
        sources=sources_list,
        additional_instruction="Allow case-insensitive matching and minor name variants."
    )

    # Publisher
    booker_publisher_leaf = evaluator.add_leaf(
        id="Booker_Publisher",
        desc="The publisher is Grove Press (or Grove Atlantic)",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="The publisher of 'Orbital' (2024 Booker Prize winner) is Grove Press (an imprint of Grove Atlantic).",
        node=booker_publisher_leaf,
        sources=sources_list,
        additional_instruction="Accept 'Grove Press' or its parent 'Grove Atlantic' when clearly indicating the same publishing group."
    )

    # Page Count
    booker_pages_leaf = evaluator.add_leaf(
        id="Booker_Page_Count",
        desc="The page count is 136 pages",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2024 Booker Prize winner 'Orbital' has 136 pages (standard edition).",
        node=booker_pages_leaf,
        sources=sources_list,
        additional_instruction="Verify the page count via publisher or authoritative bibliographic sources. Accept '136' or '136 pages'."
    )

    # Length Distinction
    booker_dist_leaf = evaluator.add_leaf(
        id="Booker_Length_Distinction",
        desc="The notable distinction is that it is the second-shortest novel to win the Booker Prize",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="Samantha Harvey's 'Orbital' is the second-shortest novel ever to win the Booker Prize.",
        node=booker_dist_leaf,
        sources=sources_list,
        additional_instruction="Check award historical context on authoritative pages. Allow minor wording variations conveying the same fact."
    )


async def verify_nyt_section(evaluator: Evaluator, parent, info: Optional[PrizeGroup]) -> None:
    section_node = evaluator.add_parallel(
        id="NYT_Bestseller_Most_Weeks_Number_One",
        desc="Information about the book that spent the most weeks at #1 on the New York Times bestseller list in 2024",
        parent=parent,
        critical=True
    )

    sources_list = _safe_sources(info)
    evaluator.add_custom_node(
        result=len(sources_list) > 0,
        id="NYT_Reference_URL",
        desc="Authoritative reference URL(s) are provided to verify the NYT bestseller information",
        parent=section_node,
        critical=True
    )

    # Title
    nyt_title_leaf = evaluator.add_leaf(
        id="NYT_Title",
        desc="The book title is 'The Women'",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="In 2024, the book that spent the most weeks at #1 on the New York Times bestseller list is titled 'The Women'.",
        node=nyt_title_leaf,
        sources=sources_list,
        additional_instruction="Ensure the statistic is specifically for the 2024 calendar year."
    )

    # Author
    nyt_author_leaf = evaluator.add_leaf(
        id="NYT_Author",
        desc="The author is Kristin Hannah",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="The author of 'The Women' (the 2024 NYT most weeks at #1) is Kristin Hannah.",
        node=nyt_author_leaf,
        sources=sources_list,
        additional_instruction="Allow case-insensitive matching."
    )

    # Publisher
    nyt_publisher_leaf = evaluator.add_leaf(
        id="NYT_Publisher",
        desc="The publisher is St. Martin's Press",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="The publisher of 'The Women' is St. Martin's Press.",
        node=nyt_publisher_leaf,
        sources=sources_list,
        additional_instruction="Accept typographic variations like St. Martin’s vs St. Martin's."
    )

    # Weeks at #1
    nyt_weeks_leaf = evaluator.add_leaf(
        id="NYT_Weeks_at_Number_One",
        desc="The number of weeks spent at #1 is 10 weeks",
        parent=section_node,
        critical=True
    )
    await evaluator.verify(
        claim="In 2024, 'The Women' spent 10 weeks at #1 on the New York Times bestseller list.",
        node=nyt_weeks_leaf,
        sources=sources_list,
        additional_instruction="Accept '10' or '10 weeks'. Ensure the count is explicitly associated with 2024, not cumulative across years."
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

    # Add a top-level node that mirrors the rubric tree's root
    top_node = evaluator.add_parallel(
        id="2024_Literary_Awards_and_Bestsellers",
        desc="Comprehensive verification of factual information about four major 2024 literary awards and bestsellers",
        parent=root,
        critical=True  # Critical parent: all its children (sections) are critical
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=LiteraryExtraction,
        extraction_name="extracted_literary_info"
    )

    # Add ground truth for transparency
    evaluator.add_ground_truth(
        gt_info=GROUND_TRUTH,
        gt_type="expected_facts"
    )

    # Verify each section using the answer's provided sources
    await verify_nba_section(evaluator, top_node, extracted.nba_fiction)
    await verify_pulitzer_section(evaluator, top_node, extracted.pulitzer_fiction)
    await verify_booker_section(evaluator, top_node, extracted.booker_prize)
    await verify_nyt_section(evaluator, top_node, extracted.nyt_most_weeks_2024)

    return evaluator.get_summary()