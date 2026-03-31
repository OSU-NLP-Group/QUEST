import asyncio
import logging
from typing import Optional, List, Dict, Tuple

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "blazer_allergy"
TASK_DESCRIPTION = """
I'm looking to buy women's blazers from ZARA, but some fabrics irritate my skin. Please find three distinct blazers (not just color variations of the same style) currently available on ZARA's website, where the combined total of Polyester (including microfiber, fleece, or metallised polyester), Nylon (including polyamide), Rayon (including viscose, modal, tencel/lyocell), and Spandex (including elastane, lycra) is less than 50% in both the outer shell and lining.

For each blazer, provide the direct ZARA link and clearly list the fabric composition percentages for the outer shell and lining.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class BlazerInfo(BaseModel):
    """Information about a single blazer from the answer."""
    zara_url: Optional[str] = None
    outer_shell_composition: Optional[str] = None
    lining_composition: Optional[str] = None


class ExtractedBlazers(BaseModel):
    """All blazers extracted from the answer."""
    blazers: List[BlazerInfo] = Field(default_factory=list)


class ProductName(BaseModel):
    """Extracted product name from a URL."""
    product_name: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_blazers() -> str:
    return """
    Extract information about all women's blazers mentioned in the answer. For each blazer, extract:

    1. The direct ZARA URL/link provided for the blazer
    2. The outer shell fabric composition exactly as stated in the answer (e.g., "72% cotton, 26% polyester, 2% other fibres")
    3. The lining fabric composition exactly as stated in the answer (e.g., "100% acetate")

    Extract the fabric composition text completely and exactly as written in the answer. 
    Do not summarize or modify the fabric composition descriptions.
    If a blazer doesn't have lining information mentioned, set lining_composition to null.
    """


def prompt_extract_product_name() -> str:
    return """
    Extract the product name of the blazer from this ZARA webpage. 
    
    Look for the main product title/name, which typically appears as a heading on the page.
    Extract only the style/model name, excluding color information if it's mentioned separately.
    
    For example:
    - If the page shows "OVERSIZED BLAZER - Black", extract "OVERSIZED BLAZER"
    - If the page shows "DOUBLE-BREASTED BLAZER", extract "DOUBLE-BREASTED BLAZER"
    - If the page shows "Textured blazer with lapel collar", extract "Textured blazer with lapel collar"
    
    Return null if no clear product name can be found.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
async def extract_product_name_from_url(
        evaluator: Evaluator,
        url: str
) -> Optional[str]:
    """Extract product name from a ZARA URL using the evaluator's extract method."""
    if not url:
        return None
    
    try:
        product_info = await evaluator.extract(
            prompt=prompt_extract_product_name(),
            template_class=ProductName,
            extraction_name=f"product_name_from_{url}",
            source=url,
            use_screenshot=True
        )
        return product_info.product_name
    except Exception as e:
        evaluator.verifier.logger.warning(f"Failed to extract product name from {url}: {e}")
        return None


async def are_products_same(
        evaluator: Evaluator,
        name1: Optional[str],
        name2: Optional[str]
) -> bool:
    """
    Use simple LLM verification to check if two product names are the same.
    Returns True if they are the SAME product.
    """
    # If either name is missing, assume they're different
    if not name1 or not name2:
        return False
    
    claim = f'"{name1}" and "{name2}" are the same ZARA blazer product.'
    
    is_same = await evaluator.verify(
        claim=claim,
        node=None,  # Don't assign to any node
        additional_instruction="Simply check if these two product names refer to the same blazer style. Ignore minor differences in capitalization or punctuation. Focus on whether they describe the same product model."
    )
    
    return is_same


async def filter_distinct_blazers(
        evaluator: Evaluator,
        blazers: List[BlazerInfo]
) -> Tuple[List[BlazerInfo], List[int], Dict[str, any]]:
    """
    Filter blazers to keep only distinct products.
    Returns (distinct_blazers, indices_of_distinct_blazers, distinctness_info)
    """
    if not blazers:
        return [], [], {"comparisons": []}
    
    # Extract product names for all blazers
    product_names = []
    for i, blazer in enumerate(blazers):
        name = await extract_product_name_from_url(evaluator, blazer.zara_url)
        product_names.append(name)
    
    # Start with the first blazer
    distinct_blazers = [blazers[0]]
    distinct_indices = [0]
    distinct_names = [product_names[0]]
    comparison_results = []
    
    # Check each subsequent blazer against all previously selected distinct blazers
    for i in range(1, len(blazers)):
        is_distinct_from_all = True
        
        for j in range(len(distinct_blazers)):
            # Use simple LLM comparison
            is_same = await are_products_same(
                evaluator,
                product_names[i],
                distinct_names[j]
            )
            
            comparison_results.append({
                "blazer_1_index": i + 1,  # 1-indexed
                "blazer_1_name": product_names[i] or "Unknown",
                "blazer_2_index": distinct_indices[j] + 1,
                "blazer_2_name": distinct_names[j] or "Unknown",
                "are_same_product": is_same
            })
            
            if is_same:
                is_distinct_from_all = False
                break
        
        # If distinct from all previously selected blazers, add it
        if is_distinct_from_all:
            distinct_blazers.append(blazers[i])
            distinct_indices.append(i)
            distinct_names.append(product_names[i])
    
    distinctness_info = {
        "total_blazers": len(blazers),
        "distinct_blazers": len(distinct_blazers),
        "duplicates_removed": len(blazers) - len(distinct_blazers),
        "product_names_extracted": [name or "Failed to extract" for name in product_names],
        "kept_indices": [i + 1 for i in distinct_indices],  # 1-indexed
        "comparison_details": comparison_results
    }
    
    return distinct_blazers, distinct_indices, distinctness_info


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_individual_blazer(
        evaluator: Evaluator,
        parent_node,
        blazer: BlazerInfo,
        blazer_index: int,
        original_index: int,
) -> None:
    """
    Verify a single blazer meets all requirements.
    
    Args:
        blazer_index: The index in the verification (0-2)
        original_index: The original index in the extracted blazers list
    """
    blazer_node = evaluator.add_parallel(
        id=f"blazer_{blazer_index + 1}",
        desc=f"Blazer {blazer_index + 1} (originally blazer {original_index + 1}) meets all requirements",
        parent=parent_node,
        critical=False,
    )

    # 1. Check for data existence
    data_exists = evaluator.add_custom_node(
        result=(blazer.zara_url is not None and blazer.zara_url.strip() != "" and 
                'zara' in blazer.zara_url.lower() and
                blazer.outer_shell_composition is not None and blazer.outer_shell_composition.strip() != ""),
        id=f"blazer_{blazer_index + 1}_data_exists",
        desc=f"Check if blazer {blazer_index + 1} has valid ZARA URL and fabric composition",
        parent=blazer_node,
        critical=True
    )

    # 2. Verify fabric compliance
    fabric_compliance_node = evaluator.add_leaf(
        id=f"blazer_{blazer_index + 1}_fabric_compliance",
        desc=f"Blazer {blazer_index + 1} fabric composition meets the <50% requirement for problematic fabrics",
        parent=blazer_node,
        critical=True,
    )

    fabric_info = f"Outer shell: {blazer.outer_shell_composition}"
    if blazer.lining_composition:
        fabric_info += f"; Lining: {blazer.lining_composition}"

    claim = f"In this fabric composition: '{fabric_info}', the combined total of Polyester (including microfiber, fleece, or metallised polyester), Nylon (including polyamide), Rayon (including viscose, modal, tencel/lyocell), and Spandex (including elastane, lycra) is less than 50% in both the outer shell and lining (if lining is present)."

    await evaluator.verify(
        claim=claim,
        node=fabric_compliance_node,
        additional_instruction="""
        Calculate the percentages of the specified problematic fabrics:
        - Polyester (including microfiber, fleece, metallised polyester)
        - Nylon (including polyamide) 
        - Rayon (including viscose, modal, tencel, lyocell)
        - Spandex (including elastane, lycra)

        For both outer shell and lining (if present), verify that the combined total of these fabrics is less than 50%.
        """
    )

    # 3. Verify URL content
    url_verification_node = evaluator.add_leaf(
        id=f"blazer_{blazer_index + 1}_url_verification",
        desc=f"Blazer {blazer_index + 1} URL leads to a ZARA blazer page and fabric information matches",
        parent=blazer_node,
        critical=True,
    )

    fabric_info_claim = f"This webpage shows a ZARA women's blazer"
    if blazer.outer_shell_composition:
        fabric_info_claim += f" with outer shell fabric composition that corresponds to: {blazer.outer_shell_composition}"
    if blazer.lining_composition:
        fabric_info_claim += f" and lining composition that corresponds to: {blazer.lining_composition}"

    await evaluator.verify(
        claim=fabric_info_claim,
        node=url_verification_node,
        sources=blazer.zara_url,
        additional_instruction="""
        Verify two things:
        1. This is a ZARA women's blazer product page
        2. The fabric composition information on the webpage corresponds to what was provided in the answer. 
           The fabric percentages should match or be consistent (exact wording may differ, but the key fabric types and percentages should align).
           Focus on verifying that the problematic fabric percentages mentioned in the answer match what's shown on the webpage. Other irrelevant details can be ignored or not included in the answer.
        """
    )


def create_placeholder_blazer_node(evaluator: Evaluator, parent_node, blazer_index: int) -> None:
    """Create a placeholder node structure for missing blazers."""
    missing_blazer_node = evaluator.add_parallel(
        id=f"blazer_{blazer_index + 1}",
        desc=f"Blazer {blazer_index + 1} meets all requirements (ZARA link, fabric compliance, URL verification)",
        parent=parent_node,
        critical=False,
    )
    
    # Add existence check that fails
    evaluator.add_custom_node(
        result=False,  # Always fails since blazer doesn't exist
        id=f"blazer_{blazer_index + 1}_data_exists",
        desc=f"Check if blazer {blazer_index + 1} has valid ZARA URL and fabric composition",
        parent=missing_blazer_node,
        critical=True
    )
    
    # Add placeholder verification nodes that will be skipped
    evaluator.add_leaf(
        id=f"blazer_{blazer_index + 1}_fabric_compliance",
        desc=f"Blazer {blazer_index + 1} fabric composition meets the <50% requirement for problematic fabrics",
        parent=missing_blazer_node,
        critical=True,
        score=0.0,
        status="skipped"
    )
    
    evaluator.add_leaf(
        id=f"blazer_{blazer_index + 1}_url_verification",
        desc=f"Blazer {blazer_index + 1} URL leads to a ZARA blazer page and fabric information matches",
        parent=missing_blazer_node,
        critical=True,
        score=0.0,
        status="skipped"
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

    # -------- 2. Extract structured info from the answer ---------------- #
    extracted_blazers = await evaluator.extract(
        prompt=prompt_extract_blazers(),
        template_class=ExtractedBlazers,
        extraction_name="blazers_info"
    )

    # -------- 3. Filter for distinct blazers using product names --------- #
    all_blazers = extracted_blazers.blazers[:3]  # Take only first 3 if more provided
    
    # Extract product names and filter for distinctness
    distinct_blazers, distinct_indices, distinctness_info = await filter_distinct_blazers(
        evaluator, 
        all_blazers
    )
    
    # Record distinctness information
    evaluator.add_custom_info(distinctness_info, "blazer_distinctness_analysis")

    # -------- 4. Build verification tree -------------------------------- #
    
    # Verify each distinct blazer (up to 3)
    for i in range(3):
        if i < len(distinct_blazers):
            await verify_individual_blazer(
                evaluator, 
                root, 
                distinct_blazers[i], 
                i,  # verification index (0-2)
                distinct_indices[i]  # original index in extracted list
            )
        else:
            # Create placeholder for missing blazers
            create_placeholder_blazer_node(evaluator, root, i)

    # -------- 5. Return structured result ------------------------------- #
    return evaluator.get_summary()