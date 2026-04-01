import asyncio
import logging
from typing import Optional, List, Dict, Any

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

from datetime import date

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "review_articles_headphones"
TASK_DESCRIPTION = """
Find review articles about the best headphones within one year and identify two headphone models that are recommended by at least 2 different articles. For each model, provide its name, links to the two review articles that recommended it, purchase links, and current prices for the model from three different purchasing sources.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ModelNames(BaseModel):
    """List of headphone model names mentioned in the answer"""
    model_names: List[str] = Field(default_factory=list)

class ReviewArticle(BaseModel):
    """Model for a review article"""
    url: Optional[str] = None
    title: Optional[str] = None
    publication_date: Optional[str] = None

class ReviewInfo(BaseModel):
    """Review articles for a specific headphone model"""
    review_articles: List[ReviewArticle] = Field(default_factory=list)

class PurchaseSource(BaseModel):
    """Model for a purchase source"""
    source_name: Optional[str] = None
    url: Optional[str] = None
    price: Optional[str] = None

class PurchaseInfo(BaseModel):
    """Purchase sources for a specific headphone model"""
    purchase_sources: List[PurchaseSource] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_model_names() -> str:
    return """
    Extract the names of exactly two headphone models that are mentioned in the answer as being recommended by at least 2 different review articles.
    
    Only extract the names of headphone models that meet these criteria. If more than two models are mentioned, extract the first two models discussed in the answer.
    
    Return a list of these model names in the order they appear in the answer.
    """

def prompt_extract_review_info(model_name: str) -> str:
    return f"""
    Extract information about the review articles that recommend the headphone model "{model_name}" as mentioned in the answer.
    
    For each review article, extract:
    1. The URL of the review article
    2. The title of the article (if mentioned)
    3. The publication date of the article (if mentioned)
    
    Only extract review articles that are explicitly linked to this specific model ("{model_name}").
    Extract all the review articles mentioned for this model, even if there are more than two.
    """

def prompt_extract_purchase_info(model_name: str) -> str:
    return f"""
    Extract information about the purchase sources for the headphone model "{model_name}" as mentioned in the answer.
    
    For each purchase source, extract:
    1. The name of the source (e.g., Amazon, Best Buy)
    2. The URL of the purchase link
    3. The price mentioned for this model at this source
    
    Only extract purchase sources that are explicitly linked to this specific model ("{model_name}").
    Extract all the purchase sources mentioned for this model, even if there are more than three.
    """

# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_recent_review_article(
    evaluator: Evaluator,
    parent_node,
    review_article: ReviewArticle,
    article_idx: int,
    model_name: str,
) -> None:
    """
    Verify that a review article is recent (within one year) and actually recommends the headphone model.
    """
    # Create parent node for this review article
    review_parent = evaluator.add_sequential(
        id=f"review_{article_idx}",
        desc=f"Verify that review article #{article_idx} for model '{model_name}' is recent and recommends the model",
        parent=parent_node,
        critical=True,
    )
    
    # 1. Check if URL exists and is valid
    url_exists = evaluator.add_custom_node(
        result=bool(review_article.url),
        id=f"review_{article_idx}_url_exists",
        desc=f"Check if URL is provided for review article #{article_idx}",
        parent=review_parent,
        critical=True
    )
    
    # 2. Verify URL is accessible and contains a review article
    url_valid_node = evaluator.add_leaf(
        id=f"review_{article_idx}_url_valid",
        desc=f"The URL contains a valid review article about headphones",
        parent=review_parent,
        critical=True,
    )
    
    await evaluator.verify(
        claim="The webpage contains a review article about the best headphones.",
        node=url_valid_node,
        sources=review_article.url,
    )
    
    # 3. Check if the article is recent (within one year)
    recent_node = evaluator.add_leaf(
        id=f"review_{article_idx}_recent",
        desc=f"The review article was published within the last year",
        parent=review_parent,
        critical=True,
    )
    
    await evaluator.verify(
        claim="The review article was published within the last year.",
        node=recent_node,
        sources=review_article.url,
        additional_instruction=f"Verify whether this article was indeed published within the last year. Today's date is {date.today()}.",
    )
    
    # 4. Check if the article actually recommends the headphone model
    recommends_node = evaluator.add_leaf(
        id=f"review_{article_idx}_recommends",
        desc=f"The review article recommends the headphone model '{model_name}'",
        parent=review_parent,
        critical=True,
    )
    
    await evaluator.verify(
        claim=f"The review article recommends the headphone model '{model_name}' as one of the best headphones.",
        node=recommends_node,
        sources=review_article.url,
        additional_instruction=f"Verify whether this article explicitly recommends the headphone model '{model_name}' as one of the best headphones or top choices. The recommendation must be positive and explicit, not just mentioning the model in passing.",
    )

async def verify_purchase_source(
    evaluator: Evaluator,
    parent_node,
    purchase_source: PurchaseSource,
    source_idx: int,
    model_name: str,
) -> None:
    """
    Verify that a purchase source is valid and has the correct price for the headphone model.
    """
    # Create parent node for this purchase source
    source_parent = evaluator.add_parallel(
        id=f"purchase_{source_idx}",
        desc=f"Verify that purchase source #{source_idx} for model '{model_name}' is valid and has the correct price",
        parent=parent_node,
        critical=False,
    )
    
    # 1. Check if URL exists
    info_exists = evaluator.add_custom_node(
        result=bool(purchase_source.url) and bool(purchase_source.price),
        id=f"purchase_{source_idx}_exists",
        desc=f"Check if URL and price is provided for purchase source #{source_idx}",
        parent=source_parent,
        critical=True
    )
    
    # 2. Verify URL is valid and accessible
    url_valid_node = evaluator.add_leaf(
        id=f"purchase_{source_idx}_url_valid",
        desc=f"The URL is valid and accessible",
        parent=source_parent,
        critical=True,
    )
    
    await evaluator.verify(
        claim=f"The webpage is a {purchase_source.source_name} purchase page for model {model_name}, and the headphone model '{model_name}' is available for purchase at the webpage.",
        node=url_valid_node,
        sources=purchase_source.url,
        additional_instruction=f"Verify whether the headphone model '{model_name}' is actually available for purchase on this website. The model should be explicitly shown as available, not just mentioned or out of stock.",
    )
    
    # 5. Verify the price matches
    price_correct_node = evaluator.add_leaf(
        id=f"purchase_{source_idx}_price_correct",
        desc=f"The price listed in the answer matches the price on the website",
        parent=source_parent,
        critical=True,
    )
    
    await evaluator.verify(
        claim=f"The price of headphone model {model_name} on {purchase_source.source_name} is {purchase_source.price}, according to the purchase page.",
        node=price_correct_node,
        sources=purchase_source.url,
        additional_instruction="""
        When comparing prices, consider that there might be slight formatting differences. 
        For example, "$299.99" and "$299" or "299 USD" should be considered matches.
        If the website shows a sale price, the answer should be considered correct if it matches either the sale price or the original price.
        If the website shows a price range (e.g., "$199-$249") and the answer price falls within that range, that should also be considered a match.
        """,
    )

async def verify_headphone_model(
    evaluator: Evaluator,
    model_name: str,
    review_info: ReviewInfo,
    purchase_info: PurchaseInfo,
    model_idx: int,
) -> None:
    """
    Verify a single headphone model meets all requirements:
    1. It has a name
    2. It is recommended by at least 2 different recent review articles
    3. It has at least 3 different purchase sources with prices
    """
    # Create the model verification node
    model_node = evaluator.add_parallel(
        id=f"model_{model_idx}",
        desc=f"Verification of headphone model '{model_name or 'Missing'}'",
        critical=False,  # Non-critical to allow partial credit between models
    )
    
    # 1. Verify the model has a name
    name_exists = evaluator.add_custom_node(
        result=bool(model_name),
        id=f"model_{model_idx}_name_exists",
        desc=f"Check if headphone model name is provided",
        parent=model_node,
        critical=True
    )
    
    # 2. Verify the model has at least 2 review articles
    reviews_node = evaluator.add_parallel(
        id=f"model_{model_idx}_reviews",
        desc=f"Headphone model '{model_name}' is recommended by at least 2 different recent review articles",
        parent=model_node,
        critical=True,
    )
    
    # Check if we have at least 2 review articles
    has_enough_reviews = evaluator.add_custom_node(
        result=len(review_info.review_articles) >= 2,
        id=f"model_{model_idx}_has_enough_reviews",
        desc=f"At least 2 review articles are provided for model '{model_name}'",
        parent=reviews_node,
        critical=True
    )
    
    # Ensure we have exactly 2 review articles to verify
    reviews_to_verify = list(review_info.review_articles[:2])
    while len(reviews_to_verify) < 2:
        reviews_to_verify.append(ReviewArticle())
    
    # Verify each review article
    for idx, review in enumerate(reviews_to_verify):
        await verify_recent_review_article(
            evaluator=evaluator,
            parent_node=reviews_node,
            review_article=review,
            article_idx=idx,
            model_name=model_name or "Missing",
        )
    
    # 3. Verify the model has at least 3 purchase sources with prices
    purchase_node = evaluator.add_parallel(
        id=f"model_{model_idx}_purchase_sources",
        desc=f"Headphone model '{model_name}' has at least 3 different purchase sources with prices",
        parent=model_node,
        critical=False,
    )
    
    # Check if we have at least 3 purchase sources
    # Ensure we have exactly 3 purchase sources to verify
    sources_to_verify = list(purchase_info.purchase_sources[:3])
    while len(sources_to_verify) < 3:
        sources_to_verify.append(PurchaseSource())
    
    # Verify each purchase source
    for idx, source in enumerate(sources_to_verify):
        await verify_purchase_source(
            evaluator=evaluator,
            parent_node=purchase_node,
            purchase_source=source,
            source_idx=idx,
            model_name=model_name or "Missing",
        )

# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict[str, Any]:
    """
    Evaluate a single answer and return a structured result dictionary.
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
        default_model=model
    )

    # -------- 2. Extract headphone models from the answer in steps ------- #
    # Step 1: Extract model names first
    model_names_info = await evaluator.extract(
        prompt=prompt_extract_model_names(),
        template_class=ModelNames,
        extraction_name="model_names"
    )
    
    # Extract detailed information for each model
    extracted_details = []
    for idx, model_name in enumerate(model_names_info.model_names[:2]):  # Process at most 2 models
        if not model_name:
            continue
            
        # Step 2a: Extract review articles for this model
        review_info = await evaluator.extract(
            prompt=prompt_extract_review_info(model_name),
            template_class=ReviewInfo,
            extraction_name=f"model_{idx}_reviews"
        )
        
        # Step 2b: Extract purchase sources for this model
        purchase_info = await evaluator.extract(
            prompt=prompt_extract_purchase_info(model_name),
            template_class=PurchaseInfo,
            extraction_name=f"model_{idx}_purchases"
        )
        
        # Store the extracted details
        extracted_details.append({
            "model_name": model_name,
            "review_info": review_info,
            "purchase_info": purchase_info
        })

    # -------- 3. Build verification tree -------------------------------- #
    # Ensure we have exactly 2 models to verify, adding empty ones if needed
    models_to_verify = []
    for idx, detail in enumerate(extracted_details[:2]):
        models_to_verify.append({
            "model_name": detail["model_name"],
            "review_info": detail["review_info"],
            "purchase_info": detail["purchase_info"],
            "model_idx": idx
        })
    
    # Add empty models if needed
    while len(models_to_verify) < 2:
        models_to_verify.append({
            "model_name": "",
            "review_info": ReviewInfo(),
            "purchase_info": PurchaseInfo(),
            "model_idx": len(models_to_verify)
        })
    
    # Verify each model (including empty ones)
    for model_data in models_to_verify:
        await verify_headphone_model(
            evaluator=evaluator,
            model_name=model_data["model_name"],
            review_info=model_data["review_info"],
            purchase_info=model_data["purchase_info"],
            model_idx=model_data["model_idx"],
        )

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()