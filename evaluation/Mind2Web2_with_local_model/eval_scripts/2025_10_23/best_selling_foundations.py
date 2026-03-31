import asyncio
import logging
from typing import Dict, List, Optional

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "best_selling_foundations"
TASK_DESCRIPTION = """
Please show me the foundations on Sephora US, sorted by best-selling, and identify the top 5 best-selling foundations from different brands. For each foundation, provide a direct link to it on its official brand website, find a concealer and highlighter from the same brand, and provide direct links to those products on the official brand websites as well.
"""

JUDGE_MODEL = "o4-mini"

# Ground truth URL for Sephora best-selling foundations
DEFAULT_SEPHORA_URL = "https://www.sephora.com/shop/foundation-makeup?sortBy=BEST_SELLING"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SephoraUrl(BaseModel):
    url: Optional[str] = None

class GroundTruthFoundation(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None
    rank: Optional[int] = None

class GroundTruthFoundations(BaseModel):
    foundations: List[GroundTruthFoundation] = Field(default_factory=list)

class FoundationMention(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None
    foundation_url: Optional[str] = None
    mentioned: bool = False

class ComplementaryProduct(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None
    url: Optional[str] = None
    found: bool = False

class ConcealerHighlighterPair(BaseModel):
    concealer: ComplementaryProduct = Field(default_factory=ComplementaryProduct)
    highlighter: ComplementaryProduct = Field(default_factory=ComplementaryProduct)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_sephora_url() -> str:
    return """
    Extract any Sephora URL mentioned in the answer text that appears to be related to best-selling foundations.
    
    Look for URLs that:
    - Contain "sephora.com" in the domain
    - Are presented in the context of showing or listing best-selling foundations
    - Are explicitly mentioned as the source for the foundation rankings
    
    Only extract URLs that are explicitly written in the answer text.
    
    Return the URL if found, or null if no such URL is mentioned in the answer.
    """

def prompt_extract_ground_truth_foundations() -> str:
    return """
    Extract the top 5 best-selling foundations from this Sephora page, in order of their ranking.
    For each foundation, extract:

    - name: the full product name
    - brand: the brand that makes the foundation
    - rank: the position in the best-selling list (1 for first, 2 for second, etc.)

    Only extract the first 5 foundations from different brands. If multiple foundations from the same brand appear, only take the highest-ranked one.
    """

def prompt_check_foundation_mentioned(foundation_name: str, brand_name: str) -> str:
    return f"""
    Extract information about a foundation by the brand '{brand_name}' that is mentioned in the answer as one of the top best-selling foundations.

    
    Be flexible in matching:
    - The exact foundation name might vary slightly from '{foundation_name}'
    - Look for any foundation from '{brand_name}' that appears to be presented as a top/best-selling foundation
    - Common brand name variations should be accepted (e.g., "Haus Labs" vs "Haus Labs by Lady Gaga")

    Return:
    - name: the exact foundation name as mentioned in the answer (only if found)
    - brand: the exact brand as mentioned in the answer (only if found)  
    - foundation_url: any URL provided for this foundation (only if found)
    - mentioned: true if a foundation from '{brand_name}' is clearly presented as one of the top best-selling foundations

    The key criterion is that the answer presents this as one of the top best-selling foundations from Sephora, not just any mention of the brand.
    """

def prompt_extract_concealer_for_brand(brand_name: str) -> str:
    return f"""
    Extract details about any concealer product from the brand '{brand_name}' mentioned in the answer.

    Return:
    - name: the concealer product name
    - brand: should be '{brand_name}'
    - url: URL to the concealer on the official brand website
    - found: true if a concealer from this brand is mentioned, false otherwise

    If multiple concealers from this brand are mentioned, extract the main one recommended.
    """

def prompt_extract_highlighter_for_brand(brand_name: str) -> str:
    return f"""
    Extract details about any highlighter product from the brand '{brand_name}' mentioned in the answer.

    Return:
    - name: the highlighter product name  
    - brand: should be '{brand_name}'
    - url: URL to the highlighter on the official brand website
    - found: true if a highlighter from this brand is mentioned, false otherwise

    If multiple highlighters from this brand are mentioned, extract the main one recommended.
    """

# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_foundation_main_task(
        evaluator: Evaluator,
        parent_node,
        gt_foundation: GroundTruthFoundation,
        foundation_mention: FoundationMention,
        position: int,
) -> None:
    """Verify the main foundation task: identification + website link."""
    main_task_node = evaluator.add_parallel(
        id=f"foundation_{position}_main",
        desc=f"Foundation #{position} ({gt_foundation.name} by {gt_foundation.brand}) main task: identification and official website link",
        parent=parent_node,
        critical=True,
    )

    # Foundation identification
    evaluator.add_custom_node(
        result=foundation_mention.mentioned,
        id=f"foundation_{position}_identified",
        desc=f"Foundation #{position} ({gt_foundation.name} by {gt_foundation.brand}) is correctly identified in the answer",
        parent=main_task_node,
        critical=True
    )

    # Check if URL exists (directly under main_task_node)
    evaluator.add_custom_node(
        result=foundation_mention.mentioned and bool(foundation_mention.foundation_url),
        id=f"foundation_{position}_url_exists",
        desc=f"Foundation #{position} has a URL provided",
        parent=main_task_node,
        critical=True
    )

    # Verify the URL (directly under main_task_node)
    website_verification_node = evaluator.add_leaf(
        id=f"foundation_{position}_website_verification",
        desc=f"Verify URL leads to foundation on official {gt_foundation.brand} website",
        parent=main_task_node,
        critical=True
    )

    claim = f"The URL leads to the foundation '{foundation_mention.name or gt_foundation.name}' on the official {gt_foundation.brand} website"
    await evaluator.verify(
        claim=claim,
        node=website_verification_node,
        sources=foundation_mention.foundation_url,
        additional_instruction=f"Verify this URL leads to the official {gt_foundation.brand} website (not a third-party retailer) and shows a foundation product that matches or is very similar to '{gt_foundation.name}'"
    )

async def verify_concealer(
        evaluator: Evaluator,
        parent_node,
        gt_foundation: GroundTruthFoundation,
        concealer: ComplementaryProduct,
        position: int,
) -> None:
    """Verify concealer from the brand."""
    concealer_node = evaluator.add_parallel(
        id=f"foundation_{position}_concealer",
        desc=f"Concealer from {gt_foundation.brand}",
        parent=parent_node,
        critical=False
    )
    
    evaluator.add_custom_node(
        result=concealer.found and bool(concealer.url),
        id=f"foundation_{position}_concealer_exists",
        desc=f"Concealer from {gt_foundation.brand} found with URL",
        parent=concealer_node,
        critical=True
    )

    concealer_verification_node = evaluator.add_leaf(
        id=f"foundation_{position}_concealer_verification",
        desc=f"Verify concealer URL for {gt_foundation.brand}",
        parent=concealer_node,
        critical=True
    )

    claim = f"The URL leads to a concealer product '{concealer.name}' on the official {gt_foundation.brand} website"
    await evaluator.verify(
        claim=claim,
        node=concealer_verification_node,
        sources=concealer.url,
        additional_instruction=f"Verify this URL leads to the official {gt_foundation.brand} website and shows a concealer product"
    )


async def verify_highlighter(
        evaluator: Evaluator,
        parent_node,
        gt_foundation: GroundTruthFoundation,
        highlighter: ComplementaryProduct,
        position: int,
) -> None:
    """Verify highlighter from the brand."""
    highlighter_node = evaluator.add_parallel(
        id=f"foundation_{position}_highlighter",
        desc=f"Highlighter from {gt_foundation.brand}",
        parent=parent_node,
        critical=False
    )
    
    evaluator.add_custom_node(
        result=highlighter.found and bool(highlighter.url),
        id=f"foundation_{position}_highlighter_exists",
        desc=f"Highlighter from {gt_foundation.brand} found with URL",
        parent=highlighter_node,
        critical=True
    )

    highlighter_verification_node = evaluator.add_leaf(
        id=f"foundation_{position}_highlighter_verification",
        desc=f"Verify highlighter URL for {gt_foundation.brand}",
        parent=highlighter_node,
        critical=True
    )

    claim = f"The URL leads to a highlighter product '{highlighter.name}' on the official {gt_foundation.brand} website"
    await evaluator.verify(
        claim=claim,
        node=highlighter_verification_node,
        sources=highlighter.url,
        additional_instruction=f"Verify this URL leads to the official {gt_foundation.brand} website and shows a highlighter product"
    )

async def verify_single_foundation(
        evaluator: Evaluator,
        parent_node,
        gt_foundation: GroundTruthFoundation,
        foundation_mention: FoundationMention,
        concealer_highlighter: ConcealerHighlighterPair,
        position: int,
) -> None:
    """Verify a complete foundation entry."""
    foundation_node = evaluator.add_parallel(
        id=f"foundation_{position}",
        desc=f"Foundation #{position}: {gt_foundation.name} by {gt_foundation.brand}",
        parent=parent_node,
        critical=False,
    )

    # Main foundation task
    await verify_foundation_main_task(
        evaluator, foundation_node, gt_foundation, foundation_mention, position
    )

    # Complementary products
    await verify_concealer(
        evaluator, foundation_node, gt_foundation, concealer_highlighter.concealer, position
    )
    
    await verify_highlighter(
        evaluator, foundation_node, gt_foundation, concealer_highlighter.highlighter, position
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
    """Evaluate the answer for best-selling foundations task."""
    
    # -------- 1. Set up evaluator ---------------------------------------- #
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
        default_model=model if model else JUDGE_MODEL
    )

    # -------- 2. Extract Sephora URL from answer or use default --------- #
    sephora_url_info = await evaluator.extract(
        prompt=prompt_extract_sephora_url(),
        template_class=SephoraUrl,
        extraction_name="sephora_url"
    )
    
    # Determine which URL to use for ground truth
    if sephora_url_info.url:
        ground_truth_url = sephora_url_info.url
        logger.info(f"Using Sephora URL from answer: {ground_truth_url}")
    else:
        ground_truth_url = DEFAULT_SEPHORA_URL
        logger.info(f"No Sephora URL found in answer, using default: {ground_truth_url}")

    # -------- 3. Extract ground truth from Sephora ---------------------- #
    ground_truth_foundations = await evaluator.extract(
        prompt=prompt_extract_ground_truth_foundations(),
        template_class=GroundTruthFoundations,
        extraction_name="ground_truth_foundations",
        source=ground_truth_url
    )

    # Ensure we have exactly 5 foundations
    gt_foundations = ground_truth_foundations.foundations[:5]
    while len(gt_foundations) < 5:
        gt_foundations.append(GroundTruthFoundation(name=None, brand=None, rank=len(gt_foundations) + 1))

    # Record ground truth info
    evaluator.add_ground_truth({
        "source_url": ground_truth_url,
        "foundations": [f.dict() for f in gt_foundations]
    }, "ground_truth_best_selling")

    # -------- 4. Extract answer information for each foundation --------- #
    foundation_mentions = []
    concealer_highlighter_pairs = []

    for gt_foundation in gt_foundations:
        if gt_foundation.name and gt_foundation.brand:
            foundation_mention = await evaluator.extract(
                prompt=prompt_check_foundation_mentioned(gt_foundation.name, gt_foundation.brand),
                template_class=FoundationMention,
                extraction_name=f"foundation_{gt_foundation.rank}_mention"
            )

            concealer = await evaluator.extract(
                prompt=prompt_extract_concealer_for_brand(gt_foundation.brand),
                template_class=ComplementaryProduct,
                extraction_name=f"concealer_{gt_foundation.brand}"
            )

            highlighter = await evaluator.extract(
                prompt=prompt_extract_highlighter_for_brand(gt_foundation.brand),
                template_class=ComplementaryProduct,
                extraction_name=f"highlighter_{gt_foundation.brand}"
            )

            concealer_highlighter_pairs.append(ConcealerHighlighterPair(
                concealer=concealer,
                highlighter=highlighter
            ))
        else:
            foundation_mention = FoundationMention()
            concealer_highlighter_pairs.append(ConcealerHighlighterPair())

        foundation_mentions.append(foundation_mention)

    # -------- 5. Build verification tree -------------------------------- #
    for i, (gt_foundation, foundation_mention, concealer_highlighter) in enumerate(
            zip(gt_foundations, foundation_mentions, concealer_highlighter_pairs), 1
    ):
        await verify_single_foundation(
            evaluator, root, gt_foundation, foundation_mention, concealer_highlighter, i
        )

    # -------- 6. Return structured result ------------------------------- #
    return evaluator.get_summary()