import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "literary_novel_2024_doubleday_nba"
TASK_DESCRIPTION = """
Identify the title and author of a literary fiction novel that meets all of the following criteria: 
(1) Published by Doubleday in 2024 in the United States, 
(2) Won the National Book Award for Fiction in 2024, 
(3) Written by an American author, 
(4) Has a page count between 300 and 350 pages (inclusive), 
(5) Won at least one other major literary prize in addition to the National Book Award. 
Provide the novel's title, author's name, and supporting reference URLs for verification.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NovelExtraction(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_year: Optional[str] = None
    publication_country: Optional[str] = None  # e.g., "United States", "U.S.", "USA"
    page_count: Optional[str] = None           # Keep as string for flexibility (e.g., "320" or "320 pages")
    genre: Optional[str] = None                # e.g., "Literary fiction", "Fiction / Literary"
    awards: List[str] = Field(default_factory=list)  # e.g., ["National Book Award for Fiction (2024)", "Pulitzer Prize (2025)"]
    reference_urls: List[str] = Field(default_factory=list)  # URLs explicitly provided in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_novel() -> str:
    return """
    From the provided answer, extract the details for the single novel the answer proposes as satisfying all constraints.
    If multiple novels are mentioned, extract only the first one presented as the main candidate.

    Required fields to extract:
    - title: the novel's title (string)
    - author: the author's full name (string)
    - publisher: the publisher stated for the relevant edition (string)
    - publication_year: the year of publication for the relevant edition (string; e.g., "2024")
    - publication_country: the country of publication for the relevant edition (string; e.g., "United States", "U.S.", "USA")
    - page_count: the page count of the relevant edition (string; e.g., "320", "320 pages", "320 pp")
    - genre: the novel's genre classification as stated (string; e.g., "Literary fiction", "Fiction / Literary")
    - awards: list of awards explicitly stated as "won" by the novel (array of strings). Include the National Book Award info if provided.
    - reference_urls: list of all URLs explicitly provided to support the claims (array of strings). Extract all valid URLs presented in the answer.

    Rules:
    - Only extract information explicitly present in the answer text.
    - If any field is missing, set it to null (for strings) or an empty list (for arrays).
    - For the URLs, include every URL that appears in the answer (including markdown links), as long as they are valid or reasonably formatted.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def verify_novel_criteria(
    evaluator: Evaluator,
    parent_node,
    novel: NovelExtraction,
) -> None:
    """
    Build verification tree nodes and run verifications for the novel criteria.
    All children under the main node are critical, matching the rubric.
    """
    # Create the rubric's main node as a critical, parallel aggregator
    main_node = evaluator.add_parallel(
        id="Novel_Identification",
        desc="Verify the response provides the required novel information and that the identified novel satisfies all specified criteria.",
        parent=parent_node,
        critical=True
    )

    # Existence checks (custom nodes, critical)
    title_exists = bool(novel.title and novel.title.strip())
    author_exists = bool(novel.author and novel.author.strip())
    urls_exist = bool(novel.reference_urls and len(novel.reference_urls) > 0)

    node_title = evaluator.add_custom_node(
        result=title_exists,
        id="Response_Provides_Title",
        desc="The response explicitly provides the novel's title.",
        parent=main_node,
        critical=True
    )

    node_author = evaluator.add_custom_node(
        result=author_exists,
        id="Response_Provides_Author_Name",
        desc="The response explicitly provides the author's name.",
        parent=main_node,
        critical=True
    )

    node_urls = evaluator.add_custom_node(
        result=urls_exist,
        id="Response_Provides_Reference_URLs",
        desc="The response includes supporting reference URL(s) sufficient to verify the key claims.",
        parent=main_node,
        critical=True
    )

    # Helper values
    title = novel.title or ""
    author = novel.author or ""
    sources = novel.reference_urls if novel.reference_urls else None
    prereq_nodes = [node_title, node_author, node_urls]

    # Leaf: Publication Year (2024)
    node_pub_year = evaluator.add_leaf(
        id="Publication_Year",
        desc="The novel was published in 2024.",
        parent=main_node,
        critical=True
    )
    claim_pub_year = f"The novel '{title}' by {author} was published in 2024 (relevant U.S. edition)."
    await evaluator.verify(
        claim=claim_pub_year,
        node=node_pub_year,
        sources=sources,
        extra_prerequisites=prereq_nodes,
        additional_instruction=(
            "Verify that at least one provided source explicitly shows a publication date or release date in 2024 "
            "for the relevant U.S. edition. If multiple editions exist, focus on the Doubleday U.S. edition."
        ),
    )

    # Leaf: Publisher (Doubleday)
    node_publisher = evaluator.add_leaf(
        id="Publisher",
        desc="The novel was published by Doubleday.",
        parent=main_node,
        critical=True
    )
    claim_publisher = (
        f"The U.S. edition of '{title}' by {author} was published by Doubleday (or its imprint branding 'Doubleday', "
        f"'Doubleday Books', or 'Knopf Doubleday Publishing Group – Doubleday')."
    )
    await evaluator.verify(
        claim=claim_publisher,
        node=node_publisher,
        sources=sources,
        extra_prerequisites=prereq_nodes,
        additional_instruction=(
            "Look for the publisher name. Accept reasonable variants that clearly indicate the Doubleday imprint, e.g., "
            "'Doubleday', 'Doubleday Books', or references to 'Knopf Doubleday Publishing Group' where Doubleday is the imprint."
        ),
    )

    # Leaf: Publication Location (United States)
    node_pub_loc = evaluator.add_leaf(
        id="Publication_Location",
        desc="The novel was published in the United States.",
        parent=main_node,
        critical=True
    )
    claim_pub_loc = f"The relevant edition of '{title}' by {author} was published in the United States (U.S.)."
    await evaluator.verify(
        claim=claim_pub_loc,
        node=node_pub_loc,
        sources=sources,
        extra_prerequisites=prereq_nodes,
        additional_instruction=(
            "Accept 'United States', 'USA', 'U.S.', 'American edition', or similar clear indicators that the edition "
            "is a U.S. publication. If a publisher page is U.S.-specific (e.g., Doubleday US site), that suffices."
        ),
    )

    # Leaf: National Book Award for Fiction 2024 (won)
    node_nba = evaluator.add_leaf(
        id="National_Book_Award",
        desc="The novel won the National Book Award for Fiction in 2024.",
        parent=main_node,
        critical=True
    )
    claim_nba = f"'{title}' by {author} won the National Book Award for Fiction in 2024."
    await evaluator.verify(
        claim=claim_nba,
        node=node_nba,
        sources=sources,
        extra_prerequisites=prereq_nodes,
        additional_instruction=(
            "Confirm explicitly that the novel is the WINNER (not longlisted, shortlisted, or finalist) of the "
            "National Book Award in the Fiction category for 2024. Accept phrasing like '2024 National Book Award—Fiction winner'."
        ),
    )

    # Leaf: Author nationality (American)
    node_author_nat = evaluator.add_leaf(
        id="Author_Nationality",
        desc="The author is American.",
        parent=main_node,
        critical=True
    )
    claim_author_nat = f"The author {author} is American (a U.S. national)."
    await evaluator.verify(
        claim=claim_author_nat,
        node=node_author_nat,
        sources=sources,
        extra_prerequisites=prereq_nodes,
        additional_instruction=(
            "Look for explicit statements like 'American author', 'American novelist', or a reliable statement indicating "
            "U.S. nationality. Place of birth alone is insufficient unless the page clearly states the author is American."
        ),
    )

    # Leaf: Page count between 300 and 350 (inclusive)
    node_pages = evaluator.add_leaf(
        id="Page_Count",
        desc="The novel has a page count between 300 and 350 pages (inclusive).",
        parent=main_node,
        critical=True
    )
    claim_pages = f"The relevant edition of '{title}' has a page count between 300 and 350 pages, inclusive."
    await evaluator.verify(
        claim=claim_pages,
        node=node_pages,
        sources=sources,
        extra_prerequisites=prereq_nodes,
        additional_instruction=(
            "Check edition details for page count (e.g., 'pages', 'pp'). If multiple editions display different page counts, "
            "prefer the U.S. Doubleday edition. Accept either hardcover or paperback if within 300–350 inclusive."
        ),
    )

    # Leaf: At least one other major award won (besides NBA)
    node_other_award = evaluator.add_leaf(
        id="Additional_Major_Award",
        desc="The novel won at least one other major literary prize besides the National Book Award.",
        parent=main_node,
        critical=True
    )
    claim_other_award = (
        f"'{title}' by {author} won at least one other major literary prize (besides the National Book Award)."
    )
    await evaluator.verify(
        claim=claim_other_award,
        node=node_other_award,
        sources=sources,
        extra_prerequisites=prereq_nodes,
        additional_instruction=(
            "Confirm the novel is a WINNER (not just shortlisted, finalist, or longlisted) of at least one other major prize. "
            "Examples of 'major' include: Pulitzer Prize (Fiction), Booker Prize, National Book Critics Circle Award (Fiction), "
            "PEN/Faulkner Award, Kirkus Prize (Fiction), Andrew Carnegie Medal for Excellence in Fiction, Women's Prize for Fiction. "
            "Equivalent internationally renowned prizes also count if clearly recognized."
        ),
    )

    # Leaf: Genre is literary fiction
    node_genre = evaluator.add_leaf(
        id="Genre",
        desc="The novel is classified as literary fiction.",
        parent=main_node,
        critical=True
    )
    claim_genre = f"'{title}' by {author} is classified as literary fiction."
    await evaluator.verify(
        claim=claim_genre,
        node=node_genre,
        sources=sources,
        extra_prerequisites=prereq_nodes,
        additional_instruction=(
            "Look for explicit genre classification such as 'literary fiction', 'fiction / literary', or descriptors like "
            "'literary'. Accept compound labels like 'literary thriller' or 'literary historical fiction' if it clearly "
            "signals the literary fiction categorization."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the literary fiction novel identification task.
    Returns a structured evaluation summary with a verification tree.
    """
    # Initialize evaluator/root
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

    # Extract the proposed novel information from the answer
    novel_info = await evaluator.extract(
        prompt=prompt_extract_novel(),
        template_class=NovelExtraction,
        extraction_name="novel_extraction",
    )

    # Record simple custom info (optional)
    evaluator.add_custom_info(
        info={
            "extracted_title": novel_info.title,
            "extracted_author": novel_info.author,
            "extracted_publisher": novel_info.publisher,
            "extracted_year": novel_info.publication_year,
            "extracted_country": novel_info.publication_country,
            "extracted_page_count": novel_info.page_count,
            "extracted_genre": novel_info.genre,
            "extracted_awards": novel_info.awards,
            "reference_urls_count": len(novel_info.reference_urls or []),
        },
        info_type="extraction_overview",
    )

    # Build and run verification nodes according to rubric
    await verify_novel_criteria(
        evaluator=evaluator,
        parent_node=root,
        novel=novel_info,
    )

    # Return final summary
    return evaluator.get_summary()