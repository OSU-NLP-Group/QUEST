import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gaming_laptops_research_v1"
TASK_DESCRIPTION = (
    "Identify four gaming laptops that are currently available for purchase from major online retailers "
    "(Best Buy, Newegg, Amazon, or official manufacturer websites). For each laptop, provide the following information: "
    "the exact processor model (including manufacturer, series, and model number, such as Intel Core i7-14650HX or AMD Ryzen 7 8845HS), "
    "the graphics card model which must be NVIDIA GeForce RTX 5060 Laptop GPU or higher (RTX 5070, RTX 5080, or RTX 5090), "
    "the display refresh rate which must be at least 144Hz, and a direct product page URL from the retailer where the laptop can be purchased."
)

# Allowed retailers guidance (domain-level hints used in the judge instructions)
ALLOWED_RETAILER_DOMAINS = [
    # Major retailers
    "bestbuy.com",
    "newegg.com",
    "amazon.com",  # Require /dp/ or /gp/product/ to be a product page
    # Common official manufacturer domains (non-exhaustive; judge should apply common sense)
    "asus.com",
    "msi.com",
    "dell.com",
    "alienware.com",  # Under Dell brand
    "lenovo.com",
    "hp.com",
    "acer.com",
    "razer.com",
    "gigabyte.com",
    "samsung.com",
    "lg.com",
]


# --------------------------------------------------------------------------- #
# Data models for information extraction                                      #
# --------------------------------------------------------------------------- #
class LaptopItem(BaseModel):
    """One laptop entry extracted from the answer."""
    name: Optional[str] = None  # Product name / model name as provided in the answer
    processor: Optional[str] = None  # e.g., "Intel Core i7-14650HX" or "AMD Ryzen 7 8845HS"
    gpu: Optional[str] = None  # e.g., "NVIDIA GeForce RTX 5070 Laptop GPU"
    refresh_rate: Optional[str] = None  # e.g., "144Hz", "165 Hz", "240hz"
    url: Optional[str] = None  # Direct product URL
    retailer: Optional[str] = None  # Optional retailer/manufacturer name if provided


class LaptopsExtraction(BaseModel):
    """Collection of all laptops extracted from the answer."""
    laptops: List[LaptopItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_laptops() -> str:
    return """
    Extract all gaming laptop entries mentioned in the answer. For each laptop, return the following fields:
    - name: The laptop's product/model name, exactly as presented in the answer.
    - processor: The exact CPU model string as presented in the answer, including manufacturer, series, and model number
                 (e.g., "Intel Core i7-14650HX" or "AMD Ryzen 7 8845HS"). If not explicitly provided, set to null.
    - gpu: The GPU model string as presented in the answer (e.g., "NVIDIA GeForce RTX 5070 Laptop GPU"). If not explicitly provided, set to null.
    - refresh_rate: The display refresh rate string as presented (e.g., "144Hz", "165 Hz", "240hz"). If not explicitly provided, set to null.
    - url: The direct product page URL for the laptop. Extract only explicit URLs shown in the answer. If missing, set to null.
    - retailer: The retailer or brand/manufacturer name if it is explicitly specified (e.g., "Best Buy", "Newegg", "Amazon", "ASUS"); otherwise null.

    Notes:
    - Return every laptop item the answer mentions; do not invent any.
    - Use the exact strings from the answer; do not try to normalize or rewrite.
    - For URLs, ensure they are valid and complete. If a URL is missing a protocol, prepend "http://" as needed.
    - It is okay if more than four laptops are extracted; we will later select the first four for verification.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _make_distinct_key(item: LaptopItem) -> str:
    # Prefer URL to determine uniqueness; if missing, fall back to name.
    url_key = _normalize_text(item.url)
    name_key = _normalize_text(item.name)
    if url_key:
        return f"url::{url_key}"
    if name_key:
        return f"name::{name_key}"
    # If both missing, create an empty key (these will be duplicates with one another)
    return "missing::"


def _allowed_retailer_instruction_text() -> str:
    return (
        "Allowed retailers are any of the following:\n"
        f"- Best Buy (bestbuy.com)\n"
        f"- Newegg (newegg.com)\n"
        f"- Amazon (amazon.com product pages only; URL path should contain '/dp/' or '/gp/product/')\n"
        f"- Official manufacturer websites (e.g., {', '.join(ALLOWED_RETAILER_DOMAINS[3:])}).\n"
        "The URL must be a direct product detail page for the specific laptop, not a category, search, review, or marketing landing page.\n"
        "Do not accept unrelated third-party retailers not listed above."
    )


def _safe_sources(url: Optional[str]) -> Optional[str]:
    url = (url or "").strip()
    return url if url else None


# --------------------------------------------------------------------------- #
# Verification logic for each laptop                                          #
# --------------------------------------------------------------------------- #
async def verify_single_laptop(
    evaluator: Evaluator,
    parent_node,
    item: LaptopItem,
    idx: int,
) -> None:
    """
    Build verification nodes for a single laptop and run verifications.
    The order: Retailer URL -> Availability -> Processor -> GPU -> Refresh Rate.
    Subsequent checks depend on Retailer URL, so they will be skipped if URL is invalid.
    """
    # Parent node for this laptop (NON-CRITICAL, to allow partial credit across laptops)
    laptop_node = evaluator.add_parallel(
        id=f"Laptop_{idx}",
        desc=f"Laptop {idx} entry meets all constraints and includes all required fields.",
        parent=parent_node,
        critical=False,
    )

    # 1) Retailer Product URL (CRITICAL leaf)
    retailer_node = evaluator.add_leaf(
        id=f"laptop_{idx}_retailer_product_url",
        desc=f"Laptop {idx} includes a direct product page URL from an allowed retailer (Best Buy, Newegg, Amazon, or official manufacturer website).",
        parent=laptop_node,
        critical=True,
    )
    retailer_claim = (
        "This webpage is a direct product detail page from an allowed retailer (Best Buy, Newegg, Amazon product page,"
        " or an official manufacturer website) for the laptop being evaluated."
    )
    await evaluator.verify(
        claim=retailer_claim,
        node=retailer_node,
        sources=_safe_sources(item.url),
        additional_instruction=(
            _allowed_retailer_instruction_text()
            + "\nIf the URL is missing or invalid, treat this as NOT SUPPORTED/false."
        ),
    )

    # 2) Availability (CRITICAL leaf)
    availability_node = evaluator.add_leaf(
        id=f"laptop_{idx}_availability",
        desc=f"Laptop {idx} is currently available for purchase (listing indicates it can be bought).",
        parent=laptop_node,
        critical=True,
    )
    availability_claim = (
        "The product page clearly indicates the laptop is currently available to purchase (e.g., 'Add to Cart', "
        "'Buy Now', 'In Stock', or similar purchasing affordances). It is not out of stock, sold out, unavailable, "
        "or preorder/coming-soon only."
    )
    await evaluator.verify(
        claim=availability_claim,
        node=availability_node,
        sources=_safe_sources(item.url),
        extra_prerequisites=[retailer_node],
        additional_instruction=(
            "Pass if the page shows an active purchase option (e.g., 'Add to Cart', 'Buy Now', price with purchase button). "
            "Fail if it says 'Out of Stock', 'Sold Out', 'Currently unavailable', 'Temporarily out of stock', 'Preorder', "
            "or similar non-available messages. If the URL is missing or invalid, mark as NOT SUPPORTED/false."
        ),
    )

    # 3) Processor Model (CRITICAL leaf)
    cpu_node = evaluator.add_leaf(
        id=f"laptop_{idx}_processor_model",
        desc=f"Laptop {idx} specifies the exact processor model including manufacturer, series, and model number.",
        parent=laptop_node,
        critical=True,
    )
    cpu_str = item.processor or ""
    cpu_claim = (
        f"The answer explicitly gives the CPU model string as '{cpu_str}'. "
        "The product page confirms this exact CPU model (manufacturer, series, and model number), "
        "allowing only minor formatting variants (case/hyphen/spacing)."
    )
    await evaluator.verify(
        claim=cpu_claim,
        node=cpu_node,
        sources=_safe_sources(item.url),
        extra_prerequisites=[retailer_node],
        additional_instruction=(
            "This check requires both: (1) the answer provides an explicit CPU model string (e.g., 'Intel Core i7-14650HX' "
            "or 'AMD Ryzen 7 8845HS'), and (2) the product page supports that exact CPU model. If the answer omitted the CPU "
            "or only gave a vague family name (e.g., 'Intel Core i7' without model number), mark as NOT SUPPORTED/false. "
            "If the URL is missing or invalid, mark as NOT SUPPORTED/false."
        ),
    )

    # 4) GPU Model and Threshold (CRITICAL leaf)
    gpu_node = evaluator.add_leaf(
        id=f"laptop_{idx}_gpu_model",
        desc=f"Laptop {idx} has an NVIDIA GeForce RTX 5060 Laptop GPU or higher (e.g., RTX 5070/5080/5090).",
        parent=laptop_node,
        critical=True,
    )
    gpu_str = item.gpu or ""
    gpu_claim = (
        f"The answer states the GPU as '{gpu_str}'. The product page confirms this GPU and it is an NVIDIA GeForce RTX 5060"
        " Laptop GPU or a higher 50-series model (e.g., RTX 5070, RTX 5080, or RTX 5090). Treat equivalent laptop GPU naming "
        "as acceptable even if the word 'Laptop' is omitted in the listing, as long as it clearly refers to a laptop GPU."
    )
    await evaluator.verify(
        claim=gpu_claim,
        node=gpu_node,
        sources=_safe_sources(item.url),
        extra_prerequisites=[retailer_node],
        additional_instruction=(
            "Pass only if the GPU on the product page is NVIDIA GeForce RTX 5060 (Laptop GPU) or higher (5070/5080/5090). "
            "If the answer did not provide any GPU model string, or the page indicates a lower/older GPU (e.g., 5050, 40xx, 30xx), "
            "mark as NOT SUPPORTED/false. If the URL is missing or invalid, mark as NOT SUPPORTED/false."
        ),
    )

    # 5) Display Refresh Rate (CRITICAL leaf)
    refresh_node = evaluator.add_leaf(
        id=f"laptop_{idx}_display_refresh_rate",
        desc=f"Laptop {idx} display refresh rate is at least 144Hz.",
        parent=laptop_node,
        critical=True,
    )
    rr_str = item.refresh_rate or ""
    refresh_claim = (
        f"The answer provides a display refresh rate '{rr_str}', and the product page confirms that the screen refresh rate is "
        "at least 144Hz (acceptable examples: 144Hz, 165Hz, 240Hz, 360Hz, 480Hz)."
    )
    await evaluator.verify(
        claim=refresh_claim,
        node=refresh_node,
        sources=_safe_sources(item.url),
        extra_prerequisites=[retailer_node],
        additional_instruction=(
            "The result should PASS only if the product page confirms a refresh rate >= 144Hz. "
            "If the answer omitted the refresh rate or the page shows <144Hz or cannot confirm, mark as NOT SUPPORTED/false. "
            "If the URL is missing or invalid, mark as NOT SUPPORTED/false."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
    Evaluate an answer for the 'Gaming Laptops Research' task.
    """
    # Initialize evaluator with root node (parallel aggregation).
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

    # 1) Extract structured laptop entries from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_laptops(),
        template_class=LaptopsExtraction,
        extraction_name="laptops_extraction",
    )

    # Record helper info for debugging
    evaluator.add_custom_info(
        {
            "total_extracted_laptops": len(extracted.laptops),
            "allowed_retailer_domains_hint": ALLOWED_RETAILER_DOMAINS,
        },
        info_type="debug",
        info_name="extraction_stats",
    )

    # 2) List-level requirements (Critical block)
    list_requirements = evaluator.add_parallel(
        id="List_Requirements",
        desc="Global list-level requirements for the set of laptops.",
        parent=root,
        critical=True,  # This gates the entire evaluation
    )

    # Determine how many we can use (first four as required)
    total_found = len(extracted.laptops)
    used_count = min(4, total_found)
    used_items: List[LaptopItem] = extracted.laptops[:used_count]

    # Provide_Exactly_Four_Laptops
    exactly_four_node = evaluator.add_custom_node(
        result=(total_found >= 4),
        id="Provide_Exactly_Four_Laptops",
        desc="The response identifies a total of four laptops (four distinct entries).",
        parent=list_requirements,
        critical=True,
    )

    # Laptops_Are_Distinct
    # Only meaningful if at least 4 were provided; otherwise this should fail as well
    distinct = False
    if total_found >= 4:
        keys = [_make_distinct_key(item) for item in used_items]
        distinct = (len(set(keys)) == 4)
    distinct_node = evaluator.add_custom_node(
        result=distinct,
        id="Laptops_Are_Distinct",
        desc="The four laptops are not duplicates of the same model/configuration repeated.",
        parent=list_requirements,
        critical=True,
    )

    # If fewer than four entries were provided, we still construct placeholders for consistent tree shape,
    # but those laptop-level checks will naturally fail/skipped due to missing URLs/fields.
    while len(used_items) < 4:
        used_items.append(LaptopItem())

    # 3) Per-laptop verification nodes (non-critical at parent, critical leaves)
    # Laptop_1
    await verify_single_laptop(evaluator, root, used_items[0], 1)
    # Laptop_2
    await verify_single_laptop(evaluator, root, used_items[1], 2)
    # Laptop_3
    await verify_single_laptop(evaluator, root, used_items[2], 3)
    # Laptop_4
    await verify_single_laptop(evaluator, root, used_items[3], 4)

    # 4) Return evaluation summary
    return evaluator.get_summary()