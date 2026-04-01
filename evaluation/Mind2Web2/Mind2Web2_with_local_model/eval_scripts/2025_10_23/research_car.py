import asyncio
import logging
from typing import Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.verification_tree import VerificationNode
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "research_car"
TASK_DESCRIPTION = """
I'm currently in the US and interested in purchasing a new car. Please help me gather the following information (for US market-only):

1. **Latest Models:**
    List all the latest sedan models for Hyundai, Mazda, and Honda, as well as their MSRP. Include the model year for each model. (no need to include trim details.)
2. **Cheapest Sedan:**
   From the list, for each brand, identify the most affordable sedan model based on the starting MSRP, and clearly state its starting MSRP.
3. **Specifications:**
    Provide at least one webpage that details the features or specifications of the identified cheapest sedan for each brand.
4. **User Reviews:**
    Supply at least two webpages per brand that display user ratings or reviews for the cheapest sedan. The reviews can pertain to different model years if needed.
5. **Comparative Posts:**
    Find three user-generated posts (excluding advertisements, professional articles, or comparison tools) that compare the cheapest sedans of these brands. The posts may refer to different model years as long as they compare the same model.
"""

BRANDS = ["Hyundai", "Mazda", "Honda"]
BRAND_URLS = {
    "Hyundai": "https://www.hyundaiusa.com/us/en/sedans",
    "Honda": "https://automobiles.honda.com/vehicles",
    "Mazda": "https://www.mazdausa.com/vehicles"
}

# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class SedanModel(BaseModel):
    """A sedan model with its year and MSRP."""
    name: Optional[str] = None
    year: Optional[str] = None
    msrp: Optional[str] = None

class BrandModels(BaseModel):
    """All sedan models for a brand."""
    models: List[SedanModel] = Field(default_factory=list)

class URLList(BaseModel):
    """List of URLs."""
    urls: List[str] = Field(default_factory=list)

class GroundTruthBrand(BaseModel):
    """Ground truth information about a brand's sedan models."""
    brand: str
    models: List[SedanModel] = Field(default_factory=list)
    cheapest_model: Optional[str] = None
    cheapest_msrp: Optional[str] = None

class GroundTruth(BaseModel):
    """Ground truth information for all brands."""
    brands: List[GroundTruthBrand] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_ground_truth_for_brand(brand: str) -> str:
    return f"""
    Extract all sedan models for {brand} from the webpage content. 
    
    For each sedan model:
    1. Extract the name of the model (e.g., "Accent", "Elantra", "Civic", "3", etc.)
    2. Extract the model year (e.g., "2024", "2025")
    3. Extract the starting MSRP (base price) for each model
    
    If any information is missing, set it to null.
    
    If no sedans are found, return an empty list.
    
    Note: Only include sedan models, not SUVs, crossovers, or other vehicle types. 
    Identify the sedans based on the webpage's categorization or description.
    """

def prompt_extract_latest_models_for_brand(brand: str) -> str:
    return f"""
    Extract all {brand} sedan models mentioned in the answer.
    
    For each sedan model, extract:
    1. The model name
    2. The model year
    3. The MSRP (price)
    
    If any information is missing for a model, set it to null.
    """

def prompt_extract_cheapest_sedan_for_brand(brand: str) -> str:
    return f"""
    Extract the cheapest {brand} sedan model identified in the answer.
    
    Extract:
    1. The model name of the cheapest {brand} sedan
    2. The model year (if provided)
    3. The MSRP (starting price) of that cheapest sedan
    
    If any information is missing, set it to null.
    """

def prompt_extract_model_urls_for_brand(brand: str) -> str:
    return f"""
    Extract all URLs from the answer that provide information about {brand} sedan models.
    
    Return a list of complete URLs in the "urls" field.
    Only include valid URLs. If no URLs are found, return an empty list.
    """

def prompt_extract_specification_links_for_brand(brand: str) -> str:
    return f"""
    Extract all URLs that link to specification webpages for the cheapest {brand} sedan model.
    
    Return a list of complete URLs in the "urls" field.
    Only include valid URLs. If no URLs are found, return an empty list.
    """

def prompt_extract_review_links_for_brand(brand: str) -> str:
    return f"""
    Extract all URLs that link to user review webpages for the cheapest {brand} sedan model.
    
    Return a list of complete URLs in the "urls" field.
    Only include valid URLs. If no URLs are found, return an empty list.
    """

def prompt_extract_comparative_posts() -> str:
    return """
    Extract all URLs to user-generated posts that compare the cheapest sedans of these brands (Hyundai, Mazda, and Honda).
    
    Return a list of complete URLs in the "urls" field.
    
    Note: Only include posts that are user-generated content (such as forum posts, discussion boards, Reddit threads, etc.)
    and exclude professional reviews, advertisements, or comparison tools.
    
    If no URLs are found, return an empty list.
    """

# --------------------------------------------------------------------------- #
# Extract ground truth data                                                   #
# --------------------------------------------------------------------------- #
async def extract_ground_truth(
    evaluator: Evaluator,
) -> GroundTruth:
    """Extract ground truth data from official brand websites."""
    ground_truth = GroundTruth()
    
    for brand, url in BRAND_URLS.items():
        # Extract sedans for the brand
        brand_sedans = await evaluator.extract(
            prompt=prompt_extract_ground_truth_for_brand(brand),
            template_class=BrandModels,
            source=url,
        )
        
        # Skip if extraction failed or no models found
        if not brand_sedans or not brand_sedans.models:
            continue
        
        # Find the cheapest model
        cheapest_model = None
        cheapest_msrp = float('inf')
        
        for model in brand_sedans.models:
            if model.msrp:
                # Extract numeric value from MSRP (remove "$", ",", etc.)
                try:
                    msrp_value = float(''.join(c for c in model.msrp if c.isdigit() or c == '.'))
                    if msrp_value < cheapest_msrp:
                        cheapest_msrp = msrp_value
                        cheapest_model = model.name
                except ValueError:
                    continue
        
        # Add to ground truth
        ground_truth_brand = GroundTruthBrand(
            brand=brand,
            models=[SedanModel(
                name=model.name,
                year=model.year,
                msrp=model.msrp
            ) for model in brand_sedans.models],
            cheapest_model=cheapest_model,
            cheapest_msrp=str(cheapest_msrp) if cheapest_msrp != float('inf') else None
        )
        
        ground_truth.brands.append(ground_truth_brand)
    
    return ground_truth


async def verify_models_list(
    evaluator: Evaluator,
    parent_node,
    brand: str,
    extracted_models: BrandModels,
    model_urls: List[str],
    ground_truth: GroundTruth,
) -> None:
    """Verify latest models for a specific brand."""
    brand_ground_truth = next((b for b in ground_truth.brands if b.brand == brand), None)

    if not brand_ground_truth or not brand_ground_truth.models:
        evaluator.add_leaf(
            id=f"{brand.lower()}_latest_models_ground_truth_missing",
            desc=f"Skipping latest {brand} sedan verification because ground truth is unavailable",
            parent=parent_node,
            critical=False,
            score=0.0,
            status="skipped"
        )
        return
    
    # Create a mapping of extracted models for easier lookup
    extracted_models_dict = {}
    if extracted_models.models:
        for model in extracted_models.models:
            if model.name:
                # Create key variations for matching
                model_key = model.name.lower().strip()
                if model.year:
                    model_with_year = f"{model_key}_{model.year}"
                    extracted_models_dict[model_with_year] = model
                extracted_models_dict[model_key] = model
    
    # Verify each ground truth model
    for gt_index, gt_model in enumerate(brand_ground_truth.models):
        # Create a node for this specific ground truth model
        gt_model_node = evaluator.add_parallel(
            id=f"{brand.lower()}_model_{gt_index}_{gt_model.name.lower().replace(' ', '_')}",
            desc=f"Verification of {brand} {gt_model.name} {gt_model.year if gt_model.year else ''}",
            parent=parent_node,
            critical=False
        )
        
        # 1. Check if the model exists in the answer
        model_exists_node = evaluator.add_leaf(
            id=f"{brand.lower()}_model_{gt_index}_existence",
            desc=f"Check if {brand} {gt_model.name} {gt_model.year if gt_model.year else ''} is mentioned in the answer",
            parent=gt_model_node,
            critical=True
        )
        
        all_mentioned_models = ", ".join([f"{model.name} ({model.year})" for model in extracted_models.models])
        # Build the claim for model existence
        existence_claim = f"{brand} {gt_model.name} ({gt_model.year}) is mentioned in the following list: {all_mentioned_models}"
        
        model_exists = await evaluator.verify(
            claim=existence_claim,
            node=model_exists_node,
            additional_instruction="Consider variations in naming, such as abbreviations, different word orders, or minor differences in model designation. For example, 'Civic' and 'Honda Civic' should be considered the same model. However, the year must be matched. For example, Civic 2024 is not the same with Civic 2025."
        )
        
        # 2. Check MSRP accuracy if the model exists and has MSRP in ground truth
        # Find the corresponding extracted model
        extracted_model = None
        
        # Try to match with year first
        if gt_model.year:
            model_key_with_year = f"{gt_model.name.lower().strip()}_{gt_model.year}"
            extracted_model = extracted_models_dict.get(model_key_with_year)
        
        # If not found with year, try without year
        if not extracted_model:
            model_key = gt_model.name.lower().strip()
            extracted_model = extracted_models_dict.get(model_key)
            
        # Also try to find by checking each extracted model
        if not extracted_model:
            for ex_model in extracted_models.models:
                if ex_model.name and gt_model.name.lower() in ex_model.name.lower():
                    if gt_model.year and ex_model.year and gt_model.year == ex_model.year:
                        extracted_model = ex_model
                        break
                    elif not gt_model.year or not ex_model.year:
                        extracted_model = ex_model
                        break
        
        if extracted_model and extracted_model.msrp:
            # Verify MSRP accuracy
            msrp_accuracy_node = evaluator.add_leaf(
                id=f"{brand.lower()}_model_{gt_index}_msrp_accuracy",
                desc=f"Verify MSRP accuracy for {brand} {gt_model.name} {gt_model.year if gt_model.year else ''}",
                parent=gt_model_node,
                critical=True
            )
            
            msrp_claim = f"The starting MSRP for the {brand} {gt_model.name}"
            if gt_model.year:
                msrp_claim += f" {gt_model.year}"
            msrp_claim += f" is {extracted_model.msrp}"
            if gt_model.msrp:
                msrp_claim += f", which matches the actual starting MSRP of {gt_model.msrp}"
            
            await evaluator.verify(
                claim=msrp_claim,
                node=msrp_accuracy_node,
                sources=model_urls if model_urls else None,
                additional_instruction=f"Verify that theMSRP matches or is very close to {extracted_model.msrp}. Allow for small variations in price representation (e.g., '$20,000' vs '$19,999' or minor differences in formatting)."
            )
        else:
            # MSRP not provided in the answer
            msrp_accuracy_node = evaluator.add_custom_node(
                result=False,
                id=f"{brand.lower()}_model_{gt_index}_msrp_accuracy",
                desc=f"MSRP not provided in the answer for {brand} {gt_model.name} {gt_model.year if gt_model.year else ''}",
                parent=gt_model_node,
                critical=True
            )

# --------------------------------------------------------------------------- #
# Brand-based verification function                                           #
# --------------------------------------------------------------------------- #
async def verify_brand_information(
    evaluator: Evaluator,
    parent_node,
    brand: str,
    extracted_models: BrandModels,
    model_urls: List[str],
    cheapest_sedan: SedanModel,
    spec_links: List[str],
    review_links: List[str],
    ground_truth: GroundTruth,
) -> List[VerificationNode]:
    """Verify all information for a specific brand."""
    # Find ground truth for this brand
    brand_ground_truth = next((b for b in ground_truth.brands if b.brand == brand), None)
    
    # ---- Verify Latest Models ----
    latest_models_node = evaluator.add_parallel(
        id=f"{brand.lower()}_latest_models",
        desc=f"Requirement 1: List of latest {brand} sedan models with year and MSRP",
        parent=parent_node
    )

    await verify_models_list(
        evaluator=evaluator,
        parent_node=latest_models_node,
        brand=brand,
        extracted_models=extracted_models,
        model_urls=model_urls,
        ground_truth=ground_truth,
    )
    
    # ---- Verify Cheapest Sedan ----
    cheapest_sedan_node = evaluator.add_parallel(
        id=f"{brand.lower()}_cheapest_sedan",
        desc=f"Requirement 2: Identification of the most affordable {brand} sedan model with its starting MSRP",
        parent=parent_node
    )

    # Check if cheapest sedan was identified with MSRP
    cheapest_exists_node = evaluator.add_custom_node(
        result=bool(cheapest_sedan and cheapest_sedan.name and cheapest_sedan.msrp),
        id=f"{brand.lower()}_cheapest_sedan_complete",
        desc=f"Check if cheapest {brand} sedan is identified with MSRP",
        parent=cheapest_sedan_node,
        critical=True
    )

    if brand_ground_truth and brand_ground_truth.cheapest_model:
        accuracy_node = evaluator.add_leaf(
            id=f"{brand.lower()}_cheapest_sedan_accuracy",
            desc=f"Verify if the identified cheapest {brand} sedan matches ground truth",
            parent=cheapest_sedan_node,
            critical=True
        )
        
        claim = f"Verify that the cheapest sedan identified, {cheapest_sedan.name if cheapest_sedan else 'N/A'}, is the same or effectively the same model {brand_ground_truth.cheapest_model} up to minor variations in naming."
        await evaluator.verify(
            claim=claim,
            node=accuracy_node
        )
    else:
        # Add a placeholder node to maintain tree structure
        accuracy_node = evaluator.add_leaf(
            id=f"{brand.lower()}_cheapest_sedan_accuracy",
            desc=f"Verify if the identified cheapest {brand} sedan matches ground truth (skipped - no ground truth available)",
            parent=cheapest_sedan_node,
            critical=False,
            score=0.0,
            status="skipped"
        )
            
    
    # ---- Verify Specification Links ----
    spec_links_node = evaluator.add_parallel(
        id=f"{brand.lower()}_specification_links",
        desc=f"Requirement 3: Webpage with specifications for the cheapest {brand} sedan",
        parent=parent_node
    )
    
    # Check if at least one specification link is provided
    spec_exists_node = evaluator.add_custom_node(
        result=bool(spec_links),
        id=f"{brand.lower()}_specification_links_existence",
        desc=f"Check if at least one specification link is provided",
        parent=spec_links_node,
        critical=True
    )
    
    # Always add validity check - let gating handle if no links
    spec_validity_node = evaluator.add_leaf(
        id=f"{brand.lower()}_specification_link_validity",
        desc=f"Verify that the URL contains specifications for the {brand} sedan",
        parent=spec_links_node,
        critical=True
    )
    
    sedan_model_name = cheapest_sedan.name if cheapest_sedan and cheapest_sedan.name else "cheapest sedan model"
    verification_claim = f"The webpage contains specifications or features for the {brand} {sedan_model_name}"
    
    await evaluator.verify(
        claim=verification_claim,
        node=spec_validity_node,
        sources=spec_links if spec_links else None,
        additional_instruction="The page should contain technical specifications, features, or detailed information about the car model, not just marketing content."
    )
    
    # ---- Verify Review Links ----
    review_links_node = evaluator.add_parallel(
        id=f"{brand.lower()}_review_links",
        desc=f"Requirement 4: Webpages with user ratings or reviews for the cheapest {brand} sedan",
        parent=parent_node
    )

    # Pad review links to ensure we always have 2
    padded_review_links = review_links.copy()
    while len(padded_review_links) < 2:
        padded_review_links.append(None)
    
    # Always create nodes for both links - let existence checks handle missing ones
    for i, review_url in enumerate(padded_review_links[:2]):
        link_node = evaluator.add_parallel(
            id=f"{brand.lower()}_review_link_{i+1}",
            desc=f"Verification of review link #{i+1} for the cheapest {brand} sedan",
            parent=review_links_node
        )
        
        # Add existence check
        link_exists_node = evaluator.add_custom_node(
            result=bool(review_url),
            id=f"{brand.lower()}_review_link_{i+1}_existence",
            desc=f"Check if review link #{i+1} URL exists",
            parent=link_node,
            critical=True
        )
        
        # Always add validity check - let gating handle if URL is None
        url_validity_node = evaluator.add_leaf(
            id=f"{brand.lower()}_review_link_{i+1}_validity",
            desc=f"Verify that the URL contains user reviews for the cheapest {brand} sedan",
            parent=link_node,
            critical=True
        )
        
        sedan_model_name = cheapest_sedan.name if cheapest_sedan and cheapest_sedan.name else "cheapest sedan model"
        verification_claim = f"The webpage contains user ratings or reviews for the {brand} {sedan_model_name}"
        
        await evaluator.verify(
            claim=verification_claim,
            node=url_validity_node,
            sources=review_url,
            additional_instruction="The page should contain actual user reviews or ratings from customers, not just professional reviews or marketing content."
        )
    return [cheapest_exists_node, accuracy_node]

# --------------------------------------------------------------------------- #
# Verification function for comparative posts                                 #
# --------------------------------------------------------------------------- #
async def verify_comparative_posts(
    evaluator: Evaluator,
    parent_node,
    comparative_posts: List[str],
    ground_truth: GroundTruth,  # Add this parameter
    cheapest_nodes: List[VerificationNode]
) -> None:
    """Verify the comparative posts requirement."""
    comparative_posts_node = evaluator.add_parallel(
        id="comparative_posts",
        desc="Requirement 5: Three user-generated posts comparing the cheapest sedans from Hyundai, Mazda, and Honda",
        parent=parent_node
    )
    
    # Pad posts to ensure we always have 3
    padded_posts = comparative_posts.copy()
    while len(padded_posts) < 3:
        padded_posts.append(None)


    # Verify each comparative post (up to three)
    for i, post_url in enumerate(padded_posts[:3]):
        post_node = evaluator.add_parallel(
            id=f"comparative_post_{i+1}",
            desc=f"Verification of comparative post #{i+1}",
            parent=comparative_posts_node
        )
        
        # Add existence check
        post_exists_node = evaluator.add_custom_node(
            result=bool(post_url),
            id=f"comparative_post_{i+1}_existence",
            desc=f"Check if comparative post #{i+1} URL exists",
            parent=post_node,
            critical=True
        )
        
        # Add verification that models mentioned are the cheapest sedans from ground truth
        # Build list of cheapest models from ground truth
        cheapest_models = []
        for brand_gt in ground_truth.brands:
            if brand_gt.cheapest_model:
                cheapest_models.append(f"{brand_gt.brand} {brand_gt.cheapest_model}")
        
        models_accuracy_node = evaluator.add_leaf(
            id=f"comparative_post_{i+1}_models_accuracy",
            desc=f"Verify that the post compares at least two of the cheapest sedan models",
            parent=post_node,
            critical=True  # Non-critical since this is accuracy verification
        )
        
        models_claim = f"The post is user-generated content (e.g., forum, discussion board, Reddit) that explicitly compares at least two of these cheapest sedan models: {', '.join(cheapest_models)}. The comparison should include models from at least two different brands."
        
        await evaluator.verify(
            claim=models_claim,
            node=models_accuracy_node,
            sources=post_url,
            additional_instruction="The page should be user-generated content like forum posts, discussion boards, Reddit threads, etc. and NOT professional reviews, advertisements, or comparison tools. Verify if the post compares at least two of the cheapest sedan models from different brands.",
            extra_prerequisites=cheapest_nodes
        )

# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Main evaluation entry point (FULLY FIXED VERSION)                           #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Safe, crash-proof evaluation:
    - All extract() results are None-protected.
    - All missing fields fallback to empty objects.
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
        default_model=model
    )

    # -------- 1. Extract ground truth safely -------------------------------- #
    logger.info("Extracting ground truth from official brand websites...")
    try:
        ground_truth = await extract_ground_truth(evaluator)
    except Exception as e:
        logger.error(f"GROUND TRUTH extraction failed: {e}")
        ground_truth = GroundTruth(brands=[])

    # -------- 2. Extract structured answer for each brand ------------------- #
    extracted_models_by_brand = {}
    model_urls_by_brand = {}
    cheapest_sedans_by_brand = {}
    spec_links_by_brand = {}
    review_links_by_brand = {}

    for brand in BRANDS:
        logger.info(f"Extracting info for brand: {brand}")

        # ---- latest models (BrandModels) ----
        models_resp = await evaluator.extract(
            prompt=prompt_extract_latest_models_for_brand(brand),
            template_class=BrandModels,
        )
        if models_resp is None:
            logger.warning(f"extract_latest_models_for_brand failed for {brand}")
            models_resp = BrandModels(models=[])
        extracted_models_by_brand[brand] = models_resp

        # ---- model URLs ----
        urls_resp = await evaluator.extract(
            prompt=prompt_extract_model_urls_for_brand(brand),
            template_class=URLList,
        )
        if urls_resp is None:
            logger.warning(f"extract_model_urls_for_brand failed for {brand}")
            model_urls_by_brand[brand] = []
        else:
            model_urls_by_brand[brand] = urls_resp.urls

        # ---- cheapest sedan ----
        cheapest_resp = await evaluator.extract(
            prompt=prompt_extract_cheapest_sedan_for_brand(brand),
            template_class=SedanModel,
        )
        if cheapest_resp is None:
            logger.warning(f"extract_cheapest_sedan_for_brand failed for {brand}")
            cheapest_resp = SedanModel(name=None, year=None, msrp=None)
        cheapest_sedans_by_brand[brand] = cheapest_resp

        # ---- specification links ----
        spec_resp = await evaluator.extract(
            prompt=prompt_extract_specification_links_for_brand(brand),
            template_class=URLList,
        )
        if spec_resp is None:
            logger.warning(f"extract_specification_links_for_brand failed for {brand}")
            spec_links_by_brand[brand] = []
        else:
            spec_links_by_brand[brand] = spec_resp.urls

        # ---- review links ----
        review_resp = await evaluator.extract(
            prompt=prompt_extract_review_links_for_brand(brand),
            template_class=URLList,
        )
        if review_resp is None:
            logger.warning(f"extract_review_links_for_brand failed for {brand}")
            review_links_by_brand[brand] = []
        else:
            review_links_by_brand[brand] = review_resp.urls

    # -------- 3. Comparative posts (safe extraction) ------------------------ #
    comparative_posts_resp = await evaluator.extract(
        prompt=prompt_extract_comparative_posts(),
        template_class=URLList,
    )
    if comparative_posts_resp is None:
        logger.warning("extract_comparative_posts failed")
        comparative_posts = []
    else:
        comparative_posts = comparative_posts_resp.urls

    # -------- 4. Add ground truth to evaluation tree ------------------------ #
    evaluator.add_ground_truth(
        {
            "brands": [gt_brand.dict() for gt_brand in ground_truth.brands]
        },
        "ground_truth_sedan_info"
    )

    # -------- 5. Brand-specific verifications ------------------------------- #
    brand_nodes = {}
    cheapest_nodes_all = []

    for brand in BRANDS:
        brand_nodes[brand] = evaluator.add_sequential(
            id=f"{brand.lower()}_information",
            desc=f"Information about {brand} sedans"
        )

        cheapest_nodes = await verify_brand_information(
            evaluator=evaluator,
            parent_node=brand_nodes[brand],
            brand=brand,
            extracted_models=extracted_models_by_brand[brand],
            model_urls=model_urls_by_brand[brand],
            cheapest_sedan=cheapest_sedans_by_brand[brand],
            spec_links=spec_links_by_brand[brand],
            review_links=review_links_by_brand[brand],
            ground_truth=ground_truth,
        )
        cheapest_nodes_all.extend(cheapest_nodes)

    # -------- 6. Comparative posts verification ----------------------------- #
    comparative_posts_parent = evaluator.add_parallel(
        id="cross_brand_comparison",
        desc="Cross-brand comparison posts"
    )

    await verify_comparative_posts(
        evaluator=evaluator,
        parent_node=comparative_posts_parent,
        comparative_posts=comparative_posts,
        ground_truth=ground_truth,
        cheapest_nodes=cheapest_nodes_all
    )

    # -------- 7. Return final structured evaluation result ------------------ #
    return evaluator.get_summary()
