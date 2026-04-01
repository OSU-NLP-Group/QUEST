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
TASK_ID = "buy_kobe_shoes_reverse_swoosh"
TASK_DESCRIPTION = """
I'm researching a custom version of the Nike Kobe 6 sneakers featuring a reverse Swoosh design (where the narrow tip of the Swoosh points toward the toe). The sneaker may be a fan-made or artist-designed custom, not an official Nike release. Please provide an article that features this specific sneaker and includes a clear photo of it.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ArticleInfo(BaseModel):
    """
    Information about the main article provided in the answer.
    
    Attributes:
        article_url: The URL of the article
        article_title: The title or description of the article
    """
    article_url: Optional[str] = None
    article_title: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_article() -> str:
    """
    Returns a prompt to extract the main article information.
    
    Returns:
        str: The extraction prompt
    """
    return """
    Extract information about the main article that discusses Nike Kobe 6 sneakers with a reverse Swoosh design.
    
    1. article_url: The URL of the main article. Extract the complete URL (including http:// or https://) 
       exactly as it appears in the answer. If multiple URLs are provided, extract the one that is 
       presented as the primary or main article. If no URL is provided, set to null.
       
    2. article_title: The title or description of the article if mentioned in the answer. 
       This could be the actual article title or how the answer describes the article.
       If not mentioned, set to null.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_article_content(
        evaluator: Evaluator,
        parent_node,
        article_url: str,
) -> None:
    """
    Verifies that the article meets all required criteria.
    
    Parameters:
        evaluator: The Evaluator instance
        parent_node: The node to attach this verification to
        article_url: The URL of the article to verify
    """
    # Create a sequential node for article verification
    article_node = evaluator.add_sequential(
        id="article_content",
        desc="Article content verification",
        parent=parent_node,
        critical=True,
    )
    
    # Verify Nike Kobe 6
    kobe_node = evaluator.add_leaf(
        id="kobe_6_featured",
        desc="Article features Nike Kobe 6 sneakers",
        parent=article_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim="The webpage shows or discusses Nike Kobe 6 sneakers or a custom version of Nike Kobe 6, identified as 'Kobe 6' or 'Kobe VI'",
        node=kobe_node,
        sources=article_url,
        additional_instruction="Check if the article specifically mentions or shows Nike Kobe 6 sneakers. Accept variations like 'Kobe VI' or custom versions clearly identified as Kobe 6."
    )
    
    # Verify reverse Swoosh design with clear photo
    swoosh_node = evaluator.add_leaf(
        id="reverse_swoosh_with_photo",
        desc="Features reverse Swoosh design with clear photo",
        parent=article_node,
        critical=True,
    )

    await evaluator.verify(
        claim="The webpage contains at least one photo of a Nike Kobe 6 sneaker that clearly shows a reverse Swoosh design, where the narrow tip of the Nike Swoosh points toward the toe of the shoe instead of toward the heel",
        node=swoosh_node,
        sources=article_url,
        additional_instruction="""Look for any photo that shows a Nike sneaker with reversed/backwards Swoosh. The key characteristic is that the narrow/pointed tip of the Swoosh points toward the toe (front) of the shoe instead of the heel (back). This may be described as 'reverse Swoosh', 'backwards Swoosh', 'inverted Swoosh', or similar terms.

The photo does NOT need to be a close-up shot - action shots, on-foot photos, or in-game photos are acceptable as long as the reverse Swoosh design is clearly visible and identifiable. The important criterion is that viewers can clearly see and identify the reverse Swoosh orientation, not the proximity or angle of the photo."""
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
    Evaluates a single answer and returns a structured result dictionary.
    
    This function orchestrates the entire evaluation process:
    1. Extracts the main article information from the answer
    2. Verifies the article meets all requirements
    3. Computes an overall score based on the verification results
    
    Parameters:
        client: The LLM client instance
        answer: The answer text to evaluate
        agent_name: Name of the agent that produced the answer
        answer_name: Name/identifier for this specific answer
        cache: Global cache for web requests
        semaphore: Concurrency control semaphore
        logger: Logging instance
        model: The LLM model to use (default: "o4-mini")
        
    Returns:
        Dict: A structured result dictionary with evaluation details
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
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

    # -------- 2. Extract article information ----------------------------- #
    article_info = await evaluator.extract(
        prompt=prompt_extract_article(),
        template_class=ArticleInfo,
        extraction_name="article_info"
    )

    # -------- 3. Build verification tree --------------------------------- #
    # Create parent node for verification
    verification_node = evaluator.add_parallel(
        id="article_verification",
        desc="Verification of article about Nike Kobe 6 with reverse Swoosh",
        critical=False,
    )
    
    # Check if article URL was provided
    url_exists = evaluator.add_custom_node(
        result=bool(article_info.article_url),
        id="article_url_provided",
        desc="Check if article URL was provided in the answer",
        parent=verification_node,
        critical=True
    )
    
    # Verify the article content
    await verify_article_content(
        evaluator=evaluator,
        parent_node=verification_node,
        article_url=article_info.article_url or "",
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()