import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dc_comics_injustice"
TASK_DESCRIPTION = """
I am a DC Comics fan. Could you find three comic books from different series written by the same author as Injustice: Gods Among Us Year One that feature Batman (excluding Injustice itself)? Please provide the links to the homepage of Injustice and these comic books on DC comic's official website.
"""

GT_AUTHOR = "Tom Taylor"
JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ComicURL(BaseModel):
    """Information about a comic book URL with description."""
    url: str
    description: str


class AllComicURLs(BaseModel):
    """Model for all extracted comic URLs from the answer."""
    injustice_url: Optional[ComicURL] = None
    batman_comic_urls: List[ComicURL] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all_urls() -> str:
    """
    Returns the prompt to extract comic URLs from the answer with intelligent categorization.

    This approach tries to identify which URLs are for which comics based on context,
    while being robust to cases where context is unclear.

    Returns:
        str: The extraction prompt for categorized comic URLs.
    """
    return """
    Extract URLs for comic books mentioned in the answer that are from DC Comics' official website.

    Based on the context and descriptions in the answer, try to identify:

    1. **injustice_url**: The URL for "Injustice: Gods Among Us Year One" (if provided)

    2. **batman_comic_urls**: URLs for Batman comics that are supposedly written by the same author as Injustice and are from different series (NOT Injustice series). 
       - If more than 3 Batman comic URLs are provided, extract only the first 3 mentioned
       - If the answer doesn't clearly identify which comics are Batman comics, extract the URLs that seem most likely to be the requested Batman comics based on context

    For each URL, provide a brief description of what comic it refers to based on the context in the answer.

    Important notes:
    - If you cannot clearly identify which URL is for Injustice from the context, set injustice_url to null
    - If you cannot identify any Batman comic URLs from the context, set batman_comic_urls to an empty list
    - Only extract URLs that appear to be from DC Comics' official website
    - Focus on quality over quantity - it's better to extract fewer, more certain URLs than many uncertain ones
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_injustice_comic(
        evaluator: Evaluator,
        parent_node,
        injustice_url: Optional[ComicURL],
) -> None:
    """
    Verifies the Injustice: Gods Among Us Year One comic URL and its author.

    This verification checks that the URL is indeed for Injustice and confirms the author.

    Parameters:
        evaluator (Evaluator): The evaluator instance.
        parent_node: The parent node to attach this verification to.
        injustice_url (Optional[ComicURL]): The identified Injustice URL to verify.
    """
    injustice_node = evaluator.add_parallel(
        id="injustice_verification",
        desc="Verification of Injustice: Gods Among Us Year One comic and its author",
        parent=parent_node,
        critical=False,
    )

    # Existence check
    existence_check = evaluator.add_custom_node(
        result=bool(injustice_url and injustice_url.url),
        id="injustice_exists",
        desc="Check if Injustice URL was provided",
        parent=injustice_node,
        critical=True
    )

    # Node 1: Verify Injustice author through URL
    author_node = evaluator.add_leaf(
        id="injustice_author_verification",
        desc=f"Verify that Injustice: Gods Among Us Year One is written by {GT_AUTHOR}",
        parent=injustice_node,
        critical=True,
    )

    author_claim = f"This comic 'Injustice: Gods Among Us Year One' is written by {GT_AUTHOR}."
    await evaluator.verify(
        claim=author_claim,
        node=author_node,
        sources=[injustice_url.url] if injustice_url else [],
        additional_instruction=f"Check if the webpage confirms that Injustice: Gods Among Us Year One is written by {GT_AUTHOR}."
    )

    # Node 2: Verify URL is from DC Comics official website
    url_node = evaluator.add_leaf(
        id="injustice_url_verification",
        desc="Verify that the URL is from DC Comics' official website",
        parent=injustice_node,
        critical=True,
    )

    url_claim = "This URL is from DC Comics' official website and is for the comic 'Injustice: Gods Among Us Year One'."
    await evaluator.verify(
        claim=url_claim,
        node=url_node,
        sources=[injustice_url.url] if injustice_url else [],
    )


async def verify_batman_comic_by_url(
        evaluator: Evaluator,
        parent_node,
        comic_url: Optional[ComicURL],
        index: int,
) -> None:
    """
    Verifies that a Batman comic meets all requirements based on its URL.

    This function verifies through the provided URL that the comic:
    1. Is written by Tom Taylor (same author as Injustice)
    2. Features Batman as a character
    3. Is not part of the Injustice series
    4. Has a valid DC Comics URL

    Parameters:
        evaluator (Evaluator): The evaluator instance.
        parent_node: The parent node to attach this verification to.
        comic_url (Optional[ComicURL]): The comic URL to verify.
        index (int): The index of this comic (1-based) for identification.
    """
    comic_node = evaluator.add_parallel(
        id=f"batman_comic_{index}_verification",
        desc=f"Verification of Batman comic #{index}: {comic_url.description if comic_url else 'not provided'}",
        parent=parent_node,
        critical=False,
    )

    # Existence check
    existence_check = evaluator.add_custom_node(
        result=bool(comic_url and comic_url.url),
        id=f"comic_{index}_exists",
        desc=f"Check if Batman comic #{index} URL was provided",
        parent=comic_node,
        critical=True
    )

    # Node 1: Verify comic author is Tom Taylor
    author_node = evaluator.add_leaf(
        id=f"comic_{index}_author",
        desc=f"Verify that this comic is written by {GT_AUTHOR}",
        parent=comic_node,
        critical=True,
    )

    author_claim = f"This comic is written by {GT_AUTHOR}."
    await evaluator.verify(
        claim=author_claim,
        node=author_node,
        sources=[comic_url.url] if comic_url else [],
        additional_instruction=f"Check if the webpage confirms that this comic is written by {GT_AUTHOR}."
    )

    # Node 2: Verify comic features Batman
    batman_node = evaluator.add_leaf(
        id=f"comic_{index}_batman",
        desc="Verify that this comic features Batman as a character",
        parent=comic_node,
        critical=True,
    )

    batman_claim = "This comic features Batman as a character."
    await evaluator.verify(
        claim=batman_claim,
        node=batman_node,
        sources=[comic_url.url] if comic_url else [],
        additional_instruction="Check if the webpage confirms that Batman appears in this comic."
    )

    # Node 3: Verify comic is not part of Injustice series
    not_injustice_node = evaluator.add_leaf(
        id=f"comic_{index}_not_injustice",
        desc="Verify that this comic is not part of the Injustice series",
        parent=comic_node,
        critical=True,
    )

    not_injustice_claim = "This comic is not part of the Injustice series."
    await evaluator.verify(
        claim=not_injustice_claim,
        node=not_injustice_node,
        sources=[comic_url.url] if comic_url else [],
        additional_instruction="Check if the comic is NOT part of the Injustice series. It should be from a different series."
    )

    # Node 4: Verify URL is from DC Comics official website
    url_node = evaluator.add_leaf(
        id=f"comic_{index}_url",
        desc="Verify that this URL is from DC Comics' official website",
        parent=comic_node,
        critical=True,
    )

    url_claim = "This URL is from DC Comics' official website."
    await evaluator.verify(
        claim=url_claim,
        node=url_node,
        sources=[comic_url.url] if comic_url else [],
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluates a single answer and returns a structured result dictionary.

    This function performs URL-based evaluation with intelligent categorization:
    1. Extract comic URLs with context-based identification
    2. Verify Injustice comic and author through its URL (1/4 score)
    3. Verify each of three Batman comics through their URLs (1/4 score each)

    The evaluation handles various cases:
    - Perfect case: 1 Injustice URL + 3 Batman comic URLs
    - Partial cases: 1+2, 1+1, 1+0, 0+X, etc.
    - Missing comics are handled with "skipped" placeholder nodes

    Parameters:
        client (Any): The LLM client for API calls.
        answer (str): The answer text to evaluate.
        agent_name (str): Name of the agent that generated the answer.
        answer_name (str): Name/identifier for this specific answer.
        cache (CacheFileSys): Global cache for storing web content.
        semaphore (asyncio.Semaphore): Semaphore for controlling concurrent API calls.
        logger (logging.Logger): Logger for diagnostic information.
        model (str): The model to use for evaluation, defaults to "o4-mini".

    Returns:
        Dict: A structured dictionary containing evaluation results.
    """
    # Set up evaluator
    evaluator = Evaluator()
    
    # Initialize evaluator
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

    # Extract comic URLs from the answer with intelligent categorization
    all_urls_response = await evaluator.extract(
        prompt=prompt_extract_all_urls(),
        template_class=AllComicURLs,
        extraction_name="all_comic_urls"
    )

    # Get extracted URLs with fallback to empty lists
    injustice_url = all_urls_response.injustice_url if all_urls_response else None
    batman_comic_urls = all_urls_response.batman_comic_urls if all_urls_response and all_urls_response.batman_comic_urls else []

    # Ensure we only take the first 3 Batman comics (as specified in task)
    batman_comic_urls = batman_comic_urls[:3]

    # Part 1: Verify Injustice comic (1/4 score)
    await verify_injustice_comic(evaluator, root, injustice_url)

    # Parts 2-4: Verify Batman comics (1/4 score each)
    # Handle cases where fewer than 3 Batman comics are provided
    for i in range(3):
        comic_url = batman_comic_urls[i] if i < len(batman_comic_urls) else None
        await verify_batman_comic_by_url(evaluator, root, comic_url, i + 1)

    # Return structured results
    return evaluator.get_summary()