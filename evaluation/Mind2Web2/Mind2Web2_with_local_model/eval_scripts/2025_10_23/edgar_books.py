import asyncio
import logging
from typing import Optional, List, Dict

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "edgar_books"
TASK_DESCRIPTION = """
Please find the Edgar Award winners for Best Novel for each of the latest ten available years. In addition, for each winning book, provide the title, author, community rating on The StoryGraph, and a link to its page on The StoryGraph.
"""

JUDGE_MODEL = "o4-mini"

# Ground truth data
GROUND_TRUTH_WINNERS = {
    "2025": {"title": "The In Crowd", "author": "Charlotte Vassell"},
    "2024": {"title": "Flags on the Bayou", "author": "James Lee Burke"},
    "2023": {"title": "Notes on an Execution", "author": "Danya Kukafka"},
    "2022": {"title": "Five Decembers", "author": "James Kestrel"},
    "2021": {"title": "Djinn Patrol on the Purple Line", "author": "Deepa Anappara"},
    "2020": {"title": "The Stranger Diaries", "author": "Elly Griffiths"},
    "2019": {"title": "Down the River Unto the Sea", "author": "Walter Mosley"},
    "2018": {"title": "Bluebird, Bluebird", "author": "Attica Locke"},
    "2017": {"title": "Before the Fall", "author": "Noah Hawley"},
    "2016": {"title": "Let Me Die in His Footsteps", "author": "Lori Roy"},
    "2015": {"title": "Mr. Mercedes", "author": "Stephen King"},
    "2014": {"title": "Ordinary Grace", "author": "William Kent Krueger"},
}

ALL_AVAILABLE_YEARS = sorted(GROUND_TRUTH_WINNERS.keys(), reverse=True)
TARGET_YEARS = ALL_AVAILABLE_YEARS[:10]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                      #
# --------------------------------------------------------------------------- #
class EdgarYearBook(BaseModel):
    """Model for the Edgar Award winning book of a specific year."""
    title: Optional[str] = None
    author: Optional[str] = None
    award_urls: List[str] = Field(default_factory=list)


class GeneralAwardUrls(BaseModel):
    """Model for general Edgar Award URLs extracted from the answer."""
    urls: List[str] = Field(default_factory=list)


class BookStorygraphInfo(BaseModel):
    """Model for a book's StoryGraph information."""
    storygraph_rating: Optional[str] = None
    storygraph_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                         #
# --------------------------------------------------------------------------- #
def prompt_extract_year_book(year: str) -> str:
    return f"""
  Extract information about the Edgar Award winner for Best Novel in the year {year} from the provided answer.

  Extract:
  1. The title of the winning book
  2. The author's name
  3. Any URLs that are provided in the answer that support the claim that this book won the Edgar Award in {year}
     (These might be links to the Edgar Awards website, news articles about the award, or any other relevant URLs)

  Return the extracted information in a structured format with the following fields:
  - title: The title of the book that won in {year} (string)
  - author: The author of the book that won in {year} (string)
  - award_urls: A list of URLs that support this book winning the award (array of strings)

  Set any missing fields to null or empty list as appropriate. Do not invent or infer information that is not explicitly stated in the answer.
  """


def prompt_extract_general_award_urls() -> str:
    return """
  Extract any URLs provided in the answer that relate to the Edgar Awards in general.
  These might be links to the Edgar Awards official website, lists of winners, or any other URLs that
  contain information about the Edgar Awards.

  Return the information in the following format:
  - urls: A list of URLs as strings (if no such URLs are provided in the answer, return an empty list)
  """


def prompt_extract_storygraph_info(year: str, title: str, author: str) -> str:
    return f"""
  Extract StoryGraph information for the book "{title}" by {author}, which won the Edgar Award for Best Novel in {year}.

  Extract:
  1. The StoryGraph community rating (if provided)
  2. The URL link to the book's page on StoryGraph (if provided)

  Return the extracted information in a structured format with the following fields:
  - storygraph_rating: The StoryGraph community rating (string)
  - storygraph_url: The URL link to the book's page on StoryGraph (string)

  Set any missing field to null. Do not invent or infer information that is not explicitly stated in the answer.
  """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_year_winner(
        evaluator: Evaluator,
        parent_node,
        year: str,
        book_info: EdgarYearBook,
        general_award_urls: GeneralAwardUrls,
) -> None:
    """
    Verify the correctness of the Edgar Award winner for a specific year.
    This is the first step for each year's verification.
    """
    # Create parent node for winner verification
    winner_parent = evaluator.add_parallel(
        id=f"winner_{year}",
        desc=f"Step 1: Correctly identify the {year} Edgar Award winner for Best Novel",
        parent=parent_node,
        critical=True,
    )
    
    # Combined existence check for book info
    book_exists = evaluator.add_custom_node(
        result=(book_info.title is not None and book_info.title.strip() != "") and 
               (book_info.author is not None and book_info.author.strip() != ""),
        id=f"book_exists_{year}",
        desc=f"Check if book title and author were provided for {year}",
        parent=winner_parent,
        critical=True
    )
    
    # Verify title and author against ground truth
    basic_info_node = evaluator.add_leaf(
        id=f"basic_info_{year}",
        desc=f"The book information correctly identifies the {year} Edgar Award winner",
        parent=winner_parent,
        critical=True,
    )
    
    gt_book = GROUND_TRUTH_WINNERS[year]
    ground_truth_claim = f"The book '{book_info.title}' by {book_info.author} matches the ground truth {year} Edgar Award winner '{gt_book['title']}' by {gt_book['author']}"
    
    await evaluator.verify(
        claim=ground_truth_claim,
        node=basic_info_node,
        additional_instruction="Check if the title and author exactly match or are very close matches (allowing for minor variations in formatting)."
    )
    
    # URL verification section
    url_verification_parent = evaluator.add_parallel(
        id=f"url_verification_parent_{year}",
        desc=f"URL verification for {year} Edgar Award winner",
        parent=winner_parent,
        critical=True,
    )
    
    # Collect all possible URLs
    verification_urls = book_info.award_urls + general_award_urls.urls

    # Existence check for URLs
    urls_exist = evaluator.add_custom_node(
        result=bool(verification_urls),
        id=f"urls_exist_{year}",
        desc=f"Check if any URLs were provided to support the {year} Edgar Award claim",
        parent=url_verification_parent,
        critical=True
    )
    
    # URL verification node
    url_node = evaluator.add_leaf(
        id=f"url_verification_{year}",
        desc=f"URLs support the {year} Edgar Award winner claim",
        parent=url_verification_parent,
        critical=True
    )
    
    info_claim = f"For the {year} Edgar Award for Best Novel, the winning book is '{book_info.title}' by {book_info.author}"
    await evaluator.verify(
        claim=info_claim,
        node=url_node,
        sources=verification_urls,
        additional_instruction=f"Verify if the claim matches the information in the provided webpages."
    )

async def verify_storygraph_info(
        evaluator: Evaluator,
        parent_node,
        year: str,
        book_info: EdgarYearBook,
        sg_info: BookStorygraphInfo,
) -> None:
    """
    Verify the StoryGraph information for a specific book.
    This is the second step for each year's verification.
    """
    # Create sequential parent for StoryGraph verification
    sg_parent = evaluator.add_sequential(
        id=f"storygraph_{year}",
        desc=f"Step 2: Provide StoryGraph information for the {year} Edgar Award winner",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )
    
    # URL verification node (first in sequence)
    url_parent = evaluator.add_parallel(
        id=f"sg_url_parent_{year}",
        desc=f"StoryGraph URL verification for {year}",
        parent=sg_parent,
        critical=True,
    )
    
    # Existence check for URL
    url_exists = evaluator.add_custom_node(
        result=(sg_info.storygraph_url is not None and 
                sg_info.storygraph_url.strip() != "" and
                "thestorygraph" in sg_info.storygraph_url.lower()),
        id=f"sg_url_exists_{year}",
        desc=f"Check if a valid StoryGraph URL was provided for {year}",
        parent=url_parent,
        critical=True
    )
    
    # URL verification
    url_node = evaluator.add_leaf(
        id=f"sg_url_{year}",
        desc=f"A valid StoryGraph URL is provided for '{book_info.title}' by {book_info.author} ({year})",
        parent=url_parent,
        critical=True,
    )
    
    url_claim = f"This is a page for the book '{book_info.title}' by {book_info.author}"
    await evaluator.verify(
        claim=url_claim,
        node=url_node,
        sources=sg_info.storygraph_url,
        additional_instruction="Check if this URL leads to a StoryGraph page for the specified book title and author. The page should clearly indicate it's for this specific book."
    )
    
    # Rating verification node (second in sequence)
    rating_parent = evaluator.add_parallel(
        id=f"sg_rating_parent_{year}",
        desc=f"StoryGraph rating verification for {year}",
        parent=sg_parent,
        critical=True,
    )
    
    # Existence check for rating
    rating_exists = evaluator.add_custom_node(
        result=(sg_info.storygraph_rating is not None and 
                sg_info.storygraph_rating.strip() != "" and
                sg_info.storygraph_url is not None),
        id=f"sg_rating_exists_{year}",
        desc=f"Check if a StoryGraph rating was provided for {year}",
        parent=rating_parent,
        critical=True
    )
    
    # Rating verification
    rating_node = evaluator.add_leaf(
        id=f"sg_rating_{year}",
        desc=f"A StoryGraph community rating is provided for '{book_info.title}' by {book_info.author} ({year})",
        parent=rating_parent,
        critical=True,
    )
    
    rating_claim = f"The book '{book_info.title}' by {book_info.author} has a StoryGraph community rating of {sg_info.storygraph_rating}"
    await evaluator.verify(
        claim=rating_claim,
        node=rating_node,
        sources=sg_info.storygraph_url,
        additional_instruction="Check if the StoryGraph community rating provided in the claim matches what is shown on this StoryGraph page. The rating is typically shown as stars or a numeric value. But, to allow minor variance since the rating changes all the time, lets allow 0.1 of variance (+-0.1)"
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: openai.AsyncAzureOpenAI,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator with parallel strategy for root
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
        default_model=model if model else JUDGE_MODEL
    )

    # -------- 2. Extract general award URLs ----------------------------- #
    general_award_urls = await evaluator.extract(
        prompt=prompt_extract_general_award_urls(),
        template_class=GeneralAwardUrls,
        extraction_name="general_award_urls"
    )

    # -------- 3. Build verification tree -------------------------------- #
    # Add ground truth info
    evaluator.add_ground_truth({
        "winners": GROUND_TRUTH_WINNERS,
        "available_years": ALL_AVAILABLE_YEARS,
        "target_years": TARGET_YEARS,
        "num_years_to_verify": len(TARGET_YEARS),
    })

    # -------- 4. Evaluate each year individually ----------------------- #
    for year in TARGET_YEARS:
        # Create a sequential node for this year's verification
        year_node = evaluator.add_sequential(
            id=f"year_{year}",
            desc=f"Verification for {year} Edgar Award winner",
            parent=root,
            critical=False,  # Non-critical to allow partial credit
        )

        # Extract information about the book for this year from the answer
        book_info = await evaluator.extract(
            prompt=prompt_extract_year_book(year),
            template_class=EdgarYearBook,
            extraction_name=f"book_info_{year}"
        )

        # Step 1: Verify the Edgar Award winner for this year
        await verify_year_winner(
            evaluator,
            year_node,
            year,
            book_info,
            general_award_urls,
        )

        # Step 2: Extract and verify StoryGraph information
        sg_info = await evaluator.extract(
            prompt=prompt_extract_storygraph_info(
                year, book_info.title or "", book_info.author or ""
            ),
            template_class=BookStorygraphInfo,
            extraction_name=f"storygraph_info_{year}"
        )

        await verify_storygraph_info(
            evaluator,
            year_node,
            year,
            book_info,
            sg_info
        )

    # -------- 5. Return structured result ------------------------------- #
    return evaluator.get_summary()
