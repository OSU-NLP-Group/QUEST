import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.verification_tree import AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "buy_stroller"
TASK_DESCRIPTION = """
Find three foldable strollers on Amazon priced between $250 and $300 with a user rating of 4.5 stars or higher, and that have real-world feedback on Reddit. For each stroller, make sure that the product itself or its series has been discussed on Reddit, and provide the product title, exact price, user rating, a direct Amazon purchase link, and a link to a Reddit discussion about the stroller or strollers from its brand.
"""
MIN_PRICE = 249  # Allowing 249 as per evaluation instructions
MAX_PRICE = 301  # Allowing 301 as per evaluation instructions
MIN_RATING = 4.5
REQUIRED_STROLLERS = 3


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class Stroller(BaseModel):
    title: Optional[str] = None
    price: Optional[str] = None  # Using string for price as instructed
    rating: Optional[str] = None  # Using string for rating as instructed
    amazon_link: Optional[str] = None
    reddit_link: Optional[str] = None


class ExtractedStrollers(BaseModel):
    strollers: List[Stroller] = Field(default_factory=list)


# Prompts for extraction
def prompt_extract_strollers() -> str:
    return """
    Extract information about all foldable strollers from Amazon mentioned in the answer. 
    For each stroller, extract:
    - title: The name or title of the stroller
    - price: The price of the stroller as mentioned in the answer (as a string)
    - rating: The user rating of the stroller (as a string)
    - amazon_link: The Amazon purchase link provided for the stroller
    - reddit_link: The Reddit discussion link about this stroller or its brand

    Only extract strollers that are explicitly mentioned as being from Amazon.
    If any field is missing for a stroller, set it to null.
    If multiple strollers are mentioned, include all of them in the extraction.
    """


# --------------------------------------------------------------------------- #
# Stroller verification functions                                             #
# --------------------------------------------------------------------------- #
async def verify_single_stroller(
        evaluator: Evaluator,
        parent_node,
        stroller: Stroller,
        index: int,
) -> None:
    """
    Verify a single stroller with all required verification steps.
    """
    # Create stroller verification parent node
    stroller_node = evaluator.add_sequential(
        id=f"stroller_{index}_verification",
        desc=f"Verification of Amazon stroller {index + 1}" + (f": {stroller.title}" if stroller.title else ""),
        parent=parent_node,
        critical=False,  # Non-critical as per instructions
    )

    # Step 1: Verify all required information exists (critical check)
    info_exists = (
        stroller.title is not None and stroller.title.strip() != "" and
        stroller.price is not None and stroller.price.strip() != "" and
        stroller.rating is not None and stroller.rating.strip() != "" and
        stroller.amazon_link is not None and stroller.amazon_link.strip() != "" and
        stroller.reddit_link is not None and stroller.reddit_link.strip() != ""
    )
    
    evaluator.add_custom_node(
        result=info_exists,
        id=f"stroller_{index}_info_verification",
        desc=f"Check if all required information is provided for Amazon stroller {index + 1}",
        parent=stroller_node,
        critical=True
    )

    # Step 2: Verify price and rating criteria
    criteria_node = evaluator.add_leaf(
        id=f"stroller_{index}_criteria_verification",
        desc=f"Verify if Amazon stroller {index + 1} meets price and rating criteria",
        parent=stroller_node,
        critical=True
    )

    # Always call verify - let the framework handle missing data
    claim = f"The stroller has a price of {stroller.price} which is between ${MIN_PRICE} and ${MAX_PRICE}, AND it has a rating of {stroller.rating} which is at least {MIN_RATING}."
    
    additional_instruction = f"""
    For the price verification:
    - Check if the price value in "{stroller.price}" is between ${MIN_PRICE} and ${MAX_PRICE}
    - Either the discounted price or original price is acceptable if there's a discount

    For the rating verification:
    - Check if the rating value in "{stroller.rating}" is at least {MIN_RATING}

    Both conditions must be met for the verification to pass.
    """

    await evaluator.verify(
        claim=claim,
        node=criteria_node,
        sources=None,  # Simple verification
        additional_instruction=additional_instruction,
    )

    # Step 3: Verify Amazon product information matches webpage
    amazon_node = evaluator.add_leaf(
        id=f"stroller_{index}_amazon_verification",
        desc=f"Verify if all information for Amazon stroller {index + 1} matches the Amazon webpage",
        parent=stroller_node,
        critical=True
    )

    # Create claim with all stroller information
    amazon_claim = f"""
    The stroller has the following information:
    - Title: {stroller.title}
    - Price: {stroller.price}
    - Rating: {stroller.rating}
    - It is a foldable stroller
    
    And, this webpage is the direct purchase link for this stroller on Amazon.
    """

    await evaluator.verify(
        claim=amazon_claim,
        node=amazon_node,
        sources=stroller.amazon_link,  # URL verification
        additional_instruction="Please check whether all the provided stroller information matches the content on the Amazon webpage. For the price, either the discounted price or the original price is acceptable if a discount is offered, but the number should accurately reflect what appears on the webpage. For foldability, check whether the title or any part of the webpage states that the stroller is foldable. If it is not explicitly mentioned, it is also acceptable to determine foldability based on the stroller's design, features, or product photos, as some foldable strollers may not specifically highlight this feature.",
    )

    # Step 4: Verify Reddit discussion is relevant
    reddit_node = evaluator.add_leaf(
        id=f"stroller_{index}_reddit_verification",
        desc=f"Verify if the Reddit discussion is about the stroller or its brand",
        parent=stroller_node,
        critical=True
    )

    reddit_claim = f"""
    This Reddit discussion is about the stroller "{stroller.title}" or about strollers from its brand.
    The discussion contains real-world feedback, user experiences, or mentions of this specific stroller model or its brand's strollers.
    """

    await evaluator.verify(
        claim=reddit_claim,
        node=reddit_node,
        sources=stroller.reddit_link,  # URL verification
        additional_instruction="Please check if this Reddit discussion mentions or discusses this specific stroller model or strollers from the same brand. The discussion should contain some form of user feedback, experience, recommendation, or mention that is relevant to this stroller or its brand. It's acceptable if the discussion is about the brand's strollers in general rather than this specific model.",
    )


async def verify_all_strollers(
        evaluator: Evaluator,
        parent_node,
        strollers: List[Stroller],
) -> None:
    """
    Verify all Amazon strollers.
    """
    strollers_node = evaluator.add_parallel(
        id="amazon_strollers_verification",
        desc="Verification of all Amazon strollers",
        parent=parent_node,
        critical=False,  # Non-critical as per instructions
    )

    # Pad the list to ensure we have exactly 3 strollers (using empty Stroller objects)
    strollers_to_verify = strollers[:REQUIRED_STROLLERS] if strollers else []
    while len(strollers_to_verify) < REQUIRED_STROLLERS:
        strollers_to_verify.append(Stroller())  # Empty stroller with all None fields

    # Verify each of the 3 strollers
    for i, stroller in enumerate(strollers_to_verify):
        await verify_single_stroller(evaluator, strollers_node, stroller, i)


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: LLMClient,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Initialize evaluator ----------------------------------- #
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        client=client,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        agent_name=agent_name,
        answer_name=answer_name,
        default_model=model,
        strategy=AggregationStrategy.PARALLEL
    )

    # -------- 2. Extract structured info -------------------------------- #
    extracted_strollers = await evaluator.extract(
        prompt=prompt_extract_strollers(),
        template_class=ExtractedStrollers,
        extraction_name="amazon_strollers",
    )

    # -------- 3. Build verification tree -------------------------------- #
    # Verify all Amazon strollers
    await verify_all_strollers(evaluator, evaluator.root, extracted_strollers.strollers)

    # -------- 4. Get final result -------------------------------------- #
    return evaluator.get_summary()