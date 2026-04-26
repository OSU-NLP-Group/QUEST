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
TASK_ID = "bestbuy_gaming_laptop_under_1000"
TASK_DESCRIPTION = (
    "I'm a college student looking for a gaming laptop at Best Buy that can handle both my schoolwork and gaming needs. "
    "Find a laptop that meets all of the following requirements: (1) Price of $1,000 or less at Best Buy, "
    "(2) NVIDIA GeForce RTX 4050 or better GPU (such as RTX 4050, 4060, 4070, 5050, 5060, 5070, or higher), "
    "(3) 16GB or more of RAM, (4) Display with 144Hz or higher refresh rate, (5) 512GB or more of SSD storage, "
    "and (6) Screen size of either 15.6 inches or 16 inches. Please provide the specific laptop model name, "
    "key specifications, and the Best Buy product page URL."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class LaptopExtraction(BaseModel):
    """
    Extract the single chosen laptop and its basic fields as stated in the answer.
    Keep all fields as strings where possible to maximize robustness to formatting.
    """
    model_name: Optional[str] = None
    bestbuy_url: Optional[str] = None
    price: Optional[str] = None
    gpu: Optional[str] = None
    ram: Optional[str] = None
    refresh_rate_hz: Optional[str] = None
    storage: Optional[str] = None
    screen_size_inches: Optional[str] = None
    other_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_laptop() -> str:
    return """
    Extract exactly one chosen laptop from the answer (the primary recommendation). If multiple laptops are mentioned,
    pick the first clearly recommended or the first one listed. Return the following fields:

    - model_name: The specific, identifiable laptop model name (e.g., "Acer Nitro 16 AN16-41" or "ASUS TUF Gaming A15").
    - bestbuy_url: The Best Buy product page URL for this specific laptop model. It must be a concrete product page URL,
      not a category or search result URL. If the answer provides multiple URLs, choose the one that best matches the model.
    - price: The quoted price in the answer for the chosen laptop (keep it as written, e.g., "$999.99" or "999").
    - gpu: The GPU string as written (e.g., "NVIDIA GeForce RTX 4050").
    - ram: The RAM string as written (e.g., "16GB DDR5").
    - refresh_rate_hz: The refresh rate string as written (e.g., "144Hz", "165Hz", or "240 Hz").
    - storage: The storage string as written (e.g., "512GB SSD", "1TB SSD").
    - screen_size_inches: The screen size string as written (e.g., "15.6-inch", "16\"", or "16 inch").
    - other_urls: Any other URLs included in the answer (if any). Exclude duplicates and ensure they are valid URLs.

    Rules:
    - Extract only what is explicitly present in the answer.
    - If a required field is missing, set it to null (or empty list for other_urls).
    - For URLs, accept both plain and markdown formats. Always output a full URL with protocol (http/https).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def verify_bestbuy_laptop(
    evaluator: Evaluator,
    parent_node,
    extracted: LaptopExtraction,
) -> None:
    """
    Build and execute the verification tree according to the rubric:
    - response_includes_model_name (critical)
    - response_includes_key_specifications (critical)
    - response_includes_bestbuy_product_url (critical; verify URL is a Best Buy product page, ideally for the stated model)
    - laptop_meets_constraints (critical; parallel checks: price<=1000, GPU>=4050, RAM>=16GB, refresh>=144Hz, storage>=512GB SSD, screen size is 15.6 or 16, listed and available)
    """

    # 1) Response includes specific model name (existence check based on extraction)
    evaluator.add_custom_node(
        result=_non_empty(extracted.model_name),
        id="response_includes_model_name",
        desc="Response provides the specific laptop model name.",
        parent=parent_node,
        critical=True,
    )

    # 2) Response includes key specifications (GPU, RAM, storage, refresh rate, and screen size)
    key_specs_present = all([
        _non_empty(extracted.gpu),
        _non_empty(extracted.ram),
        _non_empty(extracted.storage),
        _non_empty(extracted.refresh_rate_hz),
        _non_empty(extracted.screen_size_inches),
    ])
    evaluator.add_custom_node(
        result=key_specs_present,
        id="response_includes_key_specifications",
        desc="Response provides key specifications for the chosen laptop (GPU, RAM, storage, refresh rate, and screen size).",
        parent=parent_node,
        critical=True,
    )

    # 3) Response includes a valid Best Buy product URL (not a category/search page)
    url_leaf = evaluator.add_leaf(
        id="response_includes_bestbuy_product_url",
        desc="Response includes a valid Best Buy product page URL for the specific laptop model (not merely a category/search page).",
        parent=parent_node,
        critical=True,
    )
    # Build claim for URL validity; include model name if available
    if _non_empty(extracted.model_name):
        url_claim = (
            f"This URL is a legitimate Best Buy product page for the laptop model '{extracted.model_name}'. "
            f"It should not be a category or search page."
        )
    else:
        url_claim = (
            "This URL is a legitimate Best Buy product page for a specific laptop model. "
            "It should not be a category or search page."
        )

    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=extracted.bestbuy_url,
        additional_instruction=(
            "Judge whether the URL points to a concrete Best Buy product detail page (with product title, pricing, and an "
            "add-to-cart or purchase UI), not a category or search listing. If the page is not on bestbuy.com, or clearly a search/category page, it's invalid."
        ),
    )

    # 4) Laptop meets all constraints (parallel critical checks)
    constraints_parent = evaluator.add_parallel(
        id="laptop_meets_constraints",
        desc="Chosen laptop satisfies all purchase/spec constraints stated in the question/constraints.",
        parent=parent_node,
        critical=True,
    )

    # Prepare list of constraint verifications; each should depend on URL leaf (so they skip if URL invalid)
    bb_url = extracted.bestbuy_url

    # Price <= $1,000
    price_node = evaluator.add_leaf(
        id="price_constraint",
        desc="The laptop's current price at Best Buy is $1,000 or less.",
        parent=constraints_parent,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The current price shown on this Best Buy product page for the specific laptop configuration is $1,000 USD or less. "
            "Use the actual current sale/purchase price (e.g., 'Your price'/'Sale price'), not the struck-through original MSRP. "
            "If multiple configurations are shown, evaluate the one that corresponds to the specs on this page."
        ),
        node=price_node,
        sources=bb_url,
        additional_instruction=(
            "Focus on the active purchasable price on the page. If the page shows only financing/monthly amounts without a one-time price, "
            "try to identify the one-time purchase price. If the product is open-box only, consider the visible purchase price for that page."
        ),
        extra_prerequisites=[url_leaf],
    )

    # GPU requirement: RTX 4050 or better
    gpu_node = evaluator.add_leaf(
        id="gpu_requirement",
        desc="The laptop has an NVIDIA GeForce RTX 4050 or better GPU.",
        parent=constraints_parent,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The product page indicates the GPU is an NVIDIA GeForce RTX 4050 or a higher-tier model "
            "(allowed examples: RTX 4050, 4060, 4070, 4080, 4090, 5050, 5060, 5070, 5080, 5090)."
        ),
        node=gpu_node,
        sources=bb_url,
        additional_instruction=(
            "Look for the GPU in the title, specs, or key features on the Best Buy page. "
            "Any RTX 4050 or above in the 40xx or 50xx family satisfies this requirement."
        ),
        extra_prerequisites=[url_leaf],
    )

    # RAM >= 16GB
    ram_node = evaluator.add_leaf(
        id="ram_capacity",
        desc="The laptop has 16GB or more of system memory (RAM).",
        parent=constraints_parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The product page specifies system memory (RAM) of 16GB or greater.",
        node=ram_node,
        sources=bb_url,
        additional_instruction=(
            "Accept '16GB', '16 GB', '32GB', etc. If multiple memory options are presented, confirm that the sold configuration is 16GB+."
        ),
        extra_prerequisites=[url_leaf],
    )

    # Display refresh rate >= 144Hz
    refresh_node = evaluator.add_leaf(
        id="display_refresh_rate",
        desc="The laptop display refresh rate is 144Hz or higher.",
        parent=constraints_parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The product page indicates the internal display has a refresh rate of at least 144Hz (e.g., 144Hz, 165Hz, 240Hz).",
        node=refresh_node,
        sources=bb_url,
        additional_instruction=(
            "Ignore external monitor refresh rates. Check product specs/overview for the built-in screen refresh rate."
        ),
        extra_prerequisites=[url_leaf],
    )

    # Storage >= 512GB SSD
    storage_node = evaluator.add_leaf(
        id="storage_capacity",
        desc="The laptop has 512GB or more of SSD storage.",
        parent=constraints_parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The product page shows at least 512GB of SSD storage (e.g., 512GB SSD, 1TB SSD).",
        node=storage_node,
        sources=bb_url,
        additional_instruction=(
            "Confirm it's SSD storage, not HDD. If multiple storage options are listed, use the one matching the sold configuration."
        ),
        extra_prerequisites=[url_leaf],
    )

    # Screen size is 15.6" or 16"
    screen_node = evaluator.add_leaf(
        id="screen_size",
        desc="The laptop screen size is either 15.6 inches or 16 inches.",
        parent=constraints_parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The product page specifies the display size as either 15.6 inches or 16 inches.",
        node=screen_node,
        sources=bb_url,
        additional_instruction=(
            "Treat notations like 15.6\", 15.6-in, 16-in, '16 inch' as valid. Reject 15.3\", 17\", 14\", etc."
        ),
        extra_prerequisites=[url_leaf],
    )

    # Listed and available for purchase at Best Buy
    available_node = evaluator.add_leaf(
        id="listed_and_available",
        desc="The laptop is currently listed and available for purchase at Best Buy.",
        parent=constraints_parent,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The Best Buy product page shows the laptop is available for purchase (e.g., shows an 'Add to Cart' or purchase button, "
            "or indicates available shipping/pickup). It is not discontinued or completely sold out."
        ),
        node=available_node,
        sources=bb_url,
        additional_instruction=(
            "Check for availability indicators: 'Add to Cart', 'Add to Cart to See Price', 'Available for pickup/shipping'. "
            "If the page clearly states 'Sold Out', 'Unavailable', 'No Longer Available', consider it not available."
        ),
        extra_prerequisites=[url_leaf],
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Best Buy gaming laptop under $1,000 task.
    """
    # Initialize evaluator (framework root is always non-critical; we add a critical task node under it)
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

    # Extract chosen laptop details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_laptop(),
        template_class=LaptopExtraction,
        extraction_name="chosen_laptop",
    )

    # Add a top-level critical task node to reflect rubric's critical root
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Select one Best Buy laptop that satisfies all stated hardware/price/availability constraints and provide the required identifying information and link.",
        parent=root,
        critical=True,
    )

    # Build and verify nodes as per rubric
    await verify_bestbuy_laptop(evaluator, task_root, extracted)

    # Return structured summary
    return evaluator.get_summary()