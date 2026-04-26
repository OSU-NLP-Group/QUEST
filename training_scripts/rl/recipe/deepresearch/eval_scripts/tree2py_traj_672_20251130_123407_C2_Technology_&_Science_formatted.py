import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gaming_laptop_dallas_2024"
TASK_DESCRIPTION = """
Find one gaming laptop that is currently available for purchase at a major electronics retailer with a physical store location in Dallas, Texas, and meets the following minimum specifications for gaming in 2024:

- Graphics card: NVIDIA GeForce RTX 4060 or better
- RAM: At least 16GB
- Processor: Intel 12th generation or newer, OR AMD Ryzen 6000 series or newer
- Display: Refresh rate of at least 144Hz
- Storage: At least 512GB SSD

For your answer, provide:
1. The laptop manufacturer and model name/number
2. A link to the product page at the retailer's website
3. Confirmation of all five technical specifications with reference URL(s)
4. The retailer name and specific store address in Dallas, Texas
5. The current price in USD
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LaptopExtraction(BaseModel):
    # Identity and product page
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    product_page_url: Optional[str] = None

    # Retailer and Dallas store
    retailer_name: Optional[str] = None
    dallas_store_address: Optional[str] = None
    retailer_store_page_url: Optional[str] = None  # store locator or specific store page if provided

    # Hardware specifications
    gpu: Optional[str] = None
    ram: Optional[str] = None
    processor: Optional[str] = None
    display_refresh_rate: Optional[str] = None
    storage: Optional[str] = None

    # Price
    price_usd: Optional[str] = None

    # Optional supporting URLs explicitly mentioned in the answer
    spec_reference_urls: List[str] = Field(default_factory=list)
    price_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_laptop() -> str:
    return """
    Extract exactly one gaming laptop candidate from the answer (if multiple are mentioned, choose the first one).
    Return a JSON object with the following fields. If a field is missing in the answer, return null for that field.
    For URL fields, extract only URLs explicitly present in the answer (plain or markdown), do not invent.

    Fields to extract:
    - manufacturer: The laptop manufacturer/brand (e.g., ASUS, MSI, HP)
    - model: The specific model name/number (e.g., "ROG Strix G16", "15-ef2019nr")
    - product_page_url: The URL to the retailer's product page for this laptop
    - retailer_name: The retailer's name (e.g., Best Buy, Micro Center)
    - dallas_store_address: A specific physical store address in Dallas, Texas for the retailer, as provided in the answer
    - retailer_store_page_url: A URL to the retailer's store page or locator explicitly provided in the answer (if any)
    - gpu: The GPU as stated (e.g., "NVIDIA GeForce RTX 4060")
    - ram: The RAM as stated (e.g., "16GB", "32GB DDR5")
    - processor: The CPU as stated (e.g., "Intel Core i7-13700H", "AMD Ryzen 7 7840HS")
    - display_refresh_rate: The display refresh rate as stated (e.g., "144Hz", "240 Hz")
    - storage: The primary storage as stated (e.g., "512GB SSD", "1TB SSD")
    - price_usd: The current price in USD as stated in the answer (e.g., "$1,299.99", "USD 1299")
    - spec_reference_urls: An array of URLs explicitly cited as references for the specs (if any; can be the same product page URL repeated)
    - price_reference_urls: An array of URLs explicitly cited as references for the price (if any; can be the same product page URL)

    Do not infer or fabricate any data. Extract strings exactly as they appear in the answer. If a URL is missing protocol, prepend http://
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(primary_url: Optional[str], extra_urls: Optional[List[str]]) -> List[str]:
    """Combine primary URL with extra URLs, deduplicate, and drop empty values."""
    out: List[str] = []
    if primary_url and isinstance(primary_url, str) and primary_url.strip():
        out.append(primary_url.strip())
    if extra_urls:
        for u in extra_urls:
            if isinstance(u, str) and u.strip() and u.strip() not in out:
                out.append(u.strip())
    return out


def safe_str(x: Optional[str]) -> str:
    return x if isinstance(x, str) and x.strip() else ""


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_identity_and_product_page(
    evaluator: Evaluator,
    parent_node,
    data: LaptopExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Laptop_Identity_and_Product_Page",
        desc="Provide the laptop identity and a retailer product page link.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Manufacturer and Model provided (based on answer text)
    leaf_mm = evaluator.add_leaf(
        id="Manufacturer_and_Model",
        desc="Provide the laptop manufacturer and model name/number.",
        parent=node,
        critical=True
    )
    mm_claim = (
        f"The answer explicitly provides BOTH the laptop manufacturer ('{safe_str(data.manufacturer)}') "
        f"and the model ('{safe_str(data.model)}')."
    )
    await evaluator.verify(
        claim=mm_claim,
        node=leaf_mm,
        additional_instruction=(
            "Judge only by the answer text. If either manufacturer or model is missing, null, or empty, mark Incorrect. "
            "Allow minor formatting differences, but both pieces of information must be present."
        )
    )

    # Leaf: Retailer product page URL is working and is a product page
    leaf_url = evaluator.add_leaf(
        id="Retailer_Product_Page_URL",
        desc="Provide a working link to the product page on the retailer's website.",
        parent=node,
        critical=True
    )
    url_claim = (
        "This webpage is a product page on a retailer's website for the specified laptop (i.e., shows product details and purchase options)."
    )
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=data.product_page_url,
        additional_instruction=(
            "Confirm the page is on a retailer domain and appears to be a product page (e.g., contains product details and purchasing controls "
            "like Add to Cart/Buy Now). If the URL is missing or inaccessible, mark Incorrect."
        )
    )


async def build_retailer_and_dallas_store_requirement(
    evaluator: Evaluator,
    parent_node,
    data: LaptopExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Retailer_and_Dallas_Store_Requirement",
        desc="Provide the retailer identity, a specific Dallas, TX store address, confirm the retailer is a major electronics retailer, and confirm the product is currently available for purchase.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Retailer Name provided
    leaf_rn = evaluator.add_leaf(
        id="Retailer_Name",
        desc="Provide the retailer name.",
        parent=node,
        critical=True
    )
    rn_claim = f"The retailer name is explicitly provided as '{safe_str(data.retailer_name)}'."
    await evaluator.verify(
        claim=rn_claim,
        node=leaf_rn,
        additional_instruction=(
            "Judge by the answer text only. If the retailer name is missing, null, or empty, mark Incorrect."
        )
    )

    # Leaf: Dallas Store Address is a physical store in Dallas, TX
    leaf_addr = evaluator.add_leaf(
        id="Dallas_Store_Address",
        desc="Provide a specific physical store address located in Dallas, Texas for the retailer.",
        parent=node,
        critical=True
    )
    addr_claim = (
        f"The provided address '{safe_str(data.dallas_store_address)}' is a valid physical store location in Dallas, Texas "
        f"for the retailer '{safe_str(data.retailer_name)}'."
    )
    await evaluator.verify(
        claim=addr_claim,
        node=leaf_addr,
        sources=data.retailer_store_page_url,
        additional_instruction=(
            "If a store page URL is provided, verify the address belongs to a store in Dallas, TX. "
            "Otherwise, judge by the address string: It should clearly indicate 'Dallas, TX' or 'Dallas, Texas'. "
            "If the address is missing or does not appear to be in Dallas, mark Incorrect."
        )
    )

    # Leaf: Retailer is a major electronics retailer
    leaf_major = evaluator.add_leaf(
        id="Major_Electronics_Retailer",
        desc="Retailer is a major electronics retailer (as required by the prompt).",
        parent=node,
        critical=True
    )
    major_claim = (
        f"Retailer '{safe_str(data.retailer_name)}' is a major electronics retailer (large, widely recognized chain or company primarily selling consumer electronics)."
    )
    await evaluator.verify(
        claim=major_claim,
        node=leaf_major,
        sources=data.product_page_url,
        additional_instruction=(
            "Use common knowledge and context; well-known examples include Best Buy and Micro Center. "
            "If the retailer is obscure, regional-only, non-electronics-focused, or unclear, mark Incorrect."
        )
    )

    # Leaf: Product is currently available for purchase
    leaf_available = evaluator.add_leaf(
        id="Currently_Available_For_Purchase",
        desc="Confirm the laptop is currently available for purchase from the retailer (e.g., in-stock/available status shown on retailer site).",
        parent=node,
        critical=True
    )
    available_claim = (
        "The retailer's product page indicates that this laptop is currently available for purchase (e.g., in stock, Add to Cart/Buy Now enabled, available for pickup or delivery)."
    )
    await evaluator.verify(
        claim=available_claim,
        node=leaf_available,
        sources=data.product_page_url,
        additional_instruction=(
            "Look for clear availability signals like 'In Stock', 'Available', 'Add to Cart' enabled, or pickup/delivery availability. "
            "If the page shows 'Sold Out', 'Out of Stock', or otherwise unavailable, mark Incorrect."
        )
    )


async def build_hardware_specifications_with_citations(
    evaluator: Evaluator,
    parent_node,
    data: LaptopExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Hardware_Specifications_With_Citations",
        desc="Confirm all five minimum technical specifications and include reference URL(s) supporting each confirmation (URLs may all be the same product page if it contains the relevant specs).",
        parent=parent_node,
        critical=True
    )

    spec_sources = combine_sources(data.product_page_url, data.spec_reference_urls)

    # GPU Requirement
    leaf_gpu = evaluator.add_leaf(
        id="GPU_Requirement_With_Source",
        desc="Confirm GPU is NVIDIA GeForce RTX 4060 or better AND provide a supporting reference URL.",
        parent=node,
        critical=True
    )
    gpu_claim = (
        f"The laptop's GPU ('{safe_str(data.gpu)}') meets the requirement of NVIDIA GeForce RTX 4060 or better "
        f"(e.g., 4060, 4070, 4080, 4090 or 'Laptop GPU' variants)."
    )
    await evaluator.verify(
        claim=gpu_claim,
        node=leaf_gpu,
        sources=spec_sources if spec_sources else None,
        additional_instruction=(
            "Verify using the provided URL(s). If no valid URL is provided, mark Incorrect. "
            "Allow reasonable naming variations like 'GeForce RTX 4060 Laptop GPU'."
        )
    )

    # RAM Requirement
    leaf_ram = evaluator.add_leaf(
        id="RAM_Requirement_With_Source",
        desc="Confirm RAM is at least 16GB AND provide a supporting reference URL.",
        parent=node,
        critical=True
    )
    ram_claim = (
        f"The laptop's RAM ('{safe_str(data.ram)}') is at least 16GB."
    )
    await evaluator.verify(
        claim=ram_claim,
        node=leaf_ram,
        sources=spec_sources if spec_sources else None,
        additional_instruction=(
            "Verify using the provided URL(s). If no valid URL is provided, mark Incorrect. "
            "Accept '16GB', '16 GB', '32GB', '64GB', and similar; focus on capacity ≥16GB."
        )
    )

    # Processor Requirement
    leaf_cpu = evaluator.add_leaf(
        id="Processor_Requirement_With_Source",
        desc="Confirm processor is Intel 12th generation or newer OR AMD Ryzen 6000 series or newer AND provide a supporting reference URL.",
        parent=node,
        critical=True
    )
    cpu_claim = (
        f"The laptop's processor ('{safe_str(data.processor)}') satisfies Intel 12th generation or newer OR AMD Ryzen 6000 series or newer."
    )
    await evaluator.verify(
        claim=cpu_claim,
        node=leaf_cpu,
        sources=spec_sources if spec_sources else None,
        additional_instruction=(
            "Verify using the provided URL(s). If no valid URL is provided, mark Incorrect. "
            "Accept Intel 12th/13th/14th/etc. (e.g., 12700H, 13700H, 14900HX) and AMD Ryzen 6000/7000/8000 series (e.g., 6800H, 7840HS)."
        )
    )

    # Display Refresh Rate Requirement
    leaf_display = evaluator.add_leaf(
        id="Display_Requirement_With_Source",
        desc="Confirm display refresh rate is at least 144Hz AND provide a supporting reference URL.",
        parent=node,
        critical=True
    )
    display_claim = (
        f"The laptop's display refresh rate ('{safe_str(data.display_refresh_rate)}') is at least 144Hz."
    )
    await evaluator.verify(
        claim=display_claim,
        node=leaf_display,
        sources=spec_sources if spec_sources else None,
        additional_instruction=(
            "Verify using the provided URL(s). If no valid URL is provided, mark Incorrect. "
            "Accept common strings like '144Hz', '240 Hz', '165Hz'."
        )
    )

    # Storage Requirement
    leaf_storage = evaluator.add_leaf(
        id="Storage_Requirement_With_Source",
        desc="Confirm storage is at least 512GB SSD AND provide a supporting reference URL.",
        parent=node,
        critical=True
    )
    storage_claim = (
        f"The laptop's storage ('{safe_str(data.storage)}') is at least 512GB SSD."
    )
    await evaluator.verify(
        claim=storage_claim,
        node=leaf_storage,
        sources=spec_sources if spec_sources else None,
        additional_instruction=(
            "Verify using the provided URL(s). If no valid URL is provided, mark Incorrect. "
            "Accept '512GB SSD', '1TB SSD', etc.; focus on capacity ≥512GB and SSD type."
        )
    )


async def build_price_information(
    evaluator: Evaluator,
    parent_node,
    data: LaptopExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Price_Information",
        desc="Provide the current price in USD and ensure it is verifiable from the retailer's website/product page.",
        parent=parent_node,
        critical=True
    )

    price_sources = combine_sources(data.product_page_url, data.price_reference_urls)

    leaf_price = evaluator.add_leaf(
        id="Current_Price_USD_With_Source",
        desc="State the current price in USD AND provide a retailer URL where the stated current price is shown (may be the product page).",
        parent=node,
        critical=True
    )
    price_claim = (
        f"The current price shown on the retailer's product page is '{safe_str(data.price_usd)}' USD (accepting minor formatting variations)."
    )
    await evaluator.verify(
        claim=price_claim,
        node=leaf_price,
        sources=price_sources if price_sources else None,
        additional_instruction=(
            "Verify the currently displayed price on the provided URL(s). Prefer the active sale/current price over strikethrough MSRP. "
            "Accept minor formatting differences like currency symbol placement. If no valid URL is provided, mark Incorrect."
        )
    )


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
) -> Dict:
    """
    Evaluate an answer for the gaming laptop (Dallas, 2024 specs) task.
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_laptop(),
        template_class=LaptopExtraction,
        extraction_name="laptop_extraction"
    )

    # Build the verification tree under a critical main node (to gate overall pass/fail)
    main_task_node = evaluator.add_parallel(
        id="Gaming_Laptop_Research_Task",
        desc="Identify one currently purchasable gaming laptop sold by a major electronics retailer with a physical store in Dallas, TX, meeting the specified minimum hardware requirements, and report required details with verifiable sources where required.",
        parent=root,
        critical=True
    )

    # Subtrees
    await build_identity_and_product_page(evaluator, main_task_node, extracted)
    await build_retailer_and_dallas_store_requirement(evaluator, main_task_node, extracted)
    await build_hardware_specifications_with_citations(evaluator, main_task_node, extracted)
    await build_price_information(evaluator, main_task_node, extracted)

    # Return summary
    return evaluator.get_summary()