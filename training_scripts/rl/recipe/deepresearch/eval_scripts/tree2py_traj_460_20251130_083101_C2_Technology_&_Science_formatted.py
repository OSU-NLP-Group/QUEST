import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ucdenver_gaming_laptop_bfcm_2025"
TASK_DESCRIPTION = (
    "I am a prospective computer science student planning to attend the University of Colorado Denver and need to "
    "purchase a laptop that meets their CS BS/MS program requirements before starting 3000-level classes. I want to take "
    "advantage of current Black Friday/Cyber Monday 2025 deals.\n\n"
    "Find one gaming laptop currently available for purchase that meets ALL of the following specifications based on UC Denver's "
    "published requirements:\n\n"
    "- Processor: Minimum 4-core processor at 3.0 GHz or equivalent\n"
    "- RAM: At least 16GB\n"
    "- Storage: At least 500GB SSD\n"
    "- Graphics: Dedicated NVIDIA or AMD graphics card with at least 4GB VRAM\n"
    "- Operating System: Windows 11\n"
    "- Display: Minimum 1920x1080 (Full HD) resolution\n\n"
    "For the laptop you identify, provide:\n"
    "1. The exact product name and model number\n"
    "2. Verification that it meets each of the six technical specifications listed above\n"
    "3. A direct URL to the product page where it can currently be purchased\n"
    "4. The current price"
)

UC_DENVER_REQUIREMENTS = {
    "processor": "Minimum 4-core at 3.0 GHz or equivalent performance",
    "ram": "At least 16GB",
    "storage": "At least 500GB SSD",
    "graphics": "Dedicated NVIDIA or AMD GPU with at least 4GB VRAM",
    "os": "Windows 11",
    "display": "Minimum 1920x1080 (Full HD)"
}


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class LaptopSpecs(BaseModel):
    processor: Optional[str] = None
    cores: Optional[str] = None
    base_clock_ghz: Optional[str] = None
    ram: Optional[str] = None
    storage: Optional[str] = None
    gpu: Optional[str] = None
    vram: Optional[str] = None
    os: Optional[str] = None
    display_resolution: Optional[str] = None


class LaptopItem(BaseModel):
    product_name: Optional[str] = None
    model_number: Optional[str] = None
    product_url: Optional[str] = None
    current_price: Optional[str] = None
    gaming_marketing_text: Optional[str] = None
    specs: Optional[LaptopSpecs] = None


class LaptopAnswerExtraction(BaseModel):
    laptops: List[LaptopItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_laptops() -> str:
    return (
        "Extract all distinct laptop products explicitly recommended or proposed for purchase in the answer. "
        "For each laptop, return an object with the following fields:\n"
        " - product_name: The full product name as written (e.g., 'ASUS ROG Strix G16')\n"
        " - model_number: The exact model number or SKU if provided (e.g., 'G614JI-XS96')\n"
        " - product_url: A direct URL to the purchasable product page (include full URL with protocol)\n"
        " - current_price: The current price as stated (include currency symbol or text as shown)\n"
        " - gaming_marketing_text: Any text in the answer that indicates it is a 'gaming' laptop (e.g., 'Gaming', 'ROG', 'Legion', 'Nitro', 'Omen', etc.)\n"
        " - specs: an object containing:\n"
        "    • processor: CPU model text (e.g., 'Intel Core i7-13650HX')\n"
        "    • cores: core count text (e.g., '6 cores', '8-core')\n"
        "    • base_clock_ghz: a frequency text if present (e.g., '3.4 GHz', 'Boost up to 4.6 GHz')\n"
        "    • ram: RAM text (e.g., '16GB DDR5')\n"
        "    • storage: storage text (e.g., '512GB SSD', '1TB NVMe SSD')\n"
        "    • gpu: GPU text (e.g., 'NVIDIA GeForce RTX 4050')\n"
        "    • vram: VRAM text (e.g., '6GB GDDR6')\n"
        "    • os: operating system text (e.g., 'Windows 11 Home')\n"
        "    • display_resolution: resolution text (e.g., '1920x1080', '2560 x 1440')\n\n"
        "Rules:\n"
        " - Only extract information that appears in the answer text. Do not invent missing values; use null when absent.\n"
        " - If multiple laptops are mentioned, include each as a separate object in the 'laptops' array.\n"
        " - If no laptop is identified, return an empty 'laptops' array.\n"
    )


# --------------------------------------------------------------------------- #
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(x: Optional[str]) -> str:
    return x or ""


def _get_first_laptop(extraction: LaptopAnswerExtraction) -> LaptopItem:
    return extraction.laptops[0] if extraction.laptops else LaptopItem()


# --------------------------------------------------------------------------- #
# Verification Tree Construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: LaptopAnswerExtraction) -> None:
    """
    Build the verification tree and run verifications according to the rubric.
    """
    # Top-level critical group under the internal evaluator root
    main_group = evaluator.add_parallel(
        id="task_main",
        desc="Identify exactly one gaming laptop currently available that meets all UC Denver CS program requirements, with product details and price",
        parent=evaluator.root,
        critical=True
    )

    # Check exactly one laptop identified in the answer
    one_laptop_node = evaluator.add_custom_node(
        result=(len(extraction.laptops) == 1),
        id="one_laptop_only",
        desc="Response identifies exactly one laptop as the recommended option",
        parent=main_group,
        critical=True
    )

    # Use the first/only laptop for subsequent verification
    laptop = _get_first_laptop(extraction)
    product_url = laptop.product_url

    # Gaming laptop requirement: verify via product page that it is marketed as gaming
    gaming_node = evaluator.add_leaf(
        id="gaming_laptop_requirement",
        desc="The identified product is a gaming laptop (explicitly labeled/marketed as a gaming laptop)",
        parent=main_group,
        critical=True,
    )
    gaming_claim = (
        "This product page or official naming/category indicates the laptop is marketed as a gaming laptop."
    )
    await evaluator.verify(
        claim=gaming_claim,
        node=gaming_node,
        sources=product_url,
        additional_instruction=(
            "Check the product title, category breadcrumbs, or description for gaming indicators such as 'Gaming', "
            "ROG/TUF (ASUS), Legion (Lenovo), Alienware (Dell), Nitro/Predator (Acer), Omen/Victus (HP), MSI Gaming, etc. "
            "Minor naming/casing variations are acceptable."
        )
    )

    # Availability and BF/CM 2025 timeframe group (parallel, both critical)
    availability_group = evaluator.add_parallel(
        id="availability_and_sale_period",
        desc="Laptop is currently purchasable and the price/deal is from Black Friday/Cyber Monday 2025 (November 2025)",
        parent=main_group,
        critical=True
    )

    # Currently available with URL (verify via product page)
    available_node = evaluator.add_leaf(
        id="currently_available_with_url",
        desc="Provides a direct product-page URL showing the laptop is currently available to purchase",
        parent=availability_group,
        critical=True,
    )
    available_claim = (
        "This product page is a direct product-page URL and shows the laptop is currently available to purchase."
    )
    await evaluator.verify(
        claim=available_claim,
        node=available_node,
        sources=product_url,
        additional_instruction=(
            "Confirm the page is a product listing (not a generic category page) and indicates availability or a way to buy "
            "(e.g., 'Add to cart', 'Buy now', 'In stock', selectable configuration with price). If the page clearly shows "
            "unavailable or out of stock without purchase capability, this should be considered not currently purchasable."
        ),
    )

    # BF/CM 2025 timeframe (verify within the answer text)
    bfcm_node = evaluator.add_leaf(
        id="bf_cm_2025_timeframe",
        desc="States that the cited deal/price is from Black Friday/Cyber Monday 2025 (November 2025)",
        parent=availability_group,
        critical=True,
    )
    bfcm_claim = (
        "The answer explicitly states that the cited deal or price corresponds to Black Friday or Cyber Monday 2025 (November 2025)."
    )
    await evaluator.verify(
        claim=bfcm_claim,
        node=bfcm_node,
        sources=None,
        additional_instruction=(
            "Focus on the answer text. Accept formulations like 'Black Friday 2025', 'Cyber Monday 2025', 'November 2025 sales', "
            "or equivalent phrasing indicating the price/deal is from the 2025 BF/CM period."
        ),
    )

    # Product details group (parallel)
    product_details_group = evaluator.add_parallel(
        id="product_details",
        desc="Provides required identification and pricing information for the chosen laptop",
        parent=main_group,
        critical=True
    )

    # Product name and model number (verify against the product page if URL present; otherwise verify presence in answer)
    name_model_node = evaluator.add_leaf(
        id="product_name_and_model_number",
        desc="Provides the exact product name and model number",
        parent=product_details_group,
        critical=True,
    )
    name_model_claim = (
        f"The product page shows that the laptop's product name is '{_safe_str(laptop.product_name)}' "
        f"and the model number is '{_safe_str(laptop.model_number)}'. "
        "Minor formatting/casing differences are acceptable."
    )
    await evaluator.verify(
        claim=name_model_claim,
        node=name_model_node,
        sources=product_url,
        additional_instruction=(
            "Check the listing title, specifications, or SKU fields on the product page to confirm the product name and model number "
            "provided in the answer. If the URL is missing or the page does not list a model number, this should fail."
        ),
    )

    # Current price (verify against product page)
    price_node = evaluator.add_leaf(
        id="current_price",
        desc="Provides the current price of the laptop",
        parent=product_details_group,
        critical=True,
    )
    price_claim = (
        f"The current price of the laptop, as stated, is '{_safe_str(laptop.current_price)}', and this matches the product page."
    )
    await evaluator.verify(
        claim=price_claim,
        node=price_node,
        sources=product_url,
        additional_instruction=(
            "Confirm that the page shows the same price as stated in the answer. Small differences due to taxes, shipping, or "
            "currency formatting should be considered mismatches unless explicitly accounted for in the answer."
        ),
    )

    # Technical specifications group (parallel, each spec critical)
    specs_group = evaluator.add_parallel(
        id="meets_technical_specifications_with_verification",
        desc="Verifies the laptop meets each required technical specification",
        parent=main_group,
        critical=True
    )

    specs = laptop.specs or LaptopSpecs()

    # Processor requirement
    cpu_node = evaluator.add_leaf(
        id="processor_requirement",
        desc="Verification shows processor is at least 4-core and 3.0 GHz or equivalent",
        parent=specs_group,
        critical=True,
    )
    cpu_claim = (
        "The laptop's processor meets the UC Denver requirement: at least 4 cores and 3.0 GHz (base or boost/turbo acceptable as equivalent performance). "
        f"Extracted CPU spec: '{_safe_str(specs.processor)}', cores: '{_safe_str(specs.cores)}', clock: '{_safe_str(specs.base_clock_ghz)}'."
    )
    await evaluator.verify(
        claim=cpu_claim,
        node=cpu_node,
        sources=product_url,
        additional_instruction=(
            "Check the CPU details on the product page. Accept turbo/boost clock meeting or exceeding 3.0 GHz as 'equivalent' performance if base clock is not specified. "
            "If core count is at least 4 and performance frequency (base or boost) is ≥ 3.0 GHz, pass."
        ),
    )

    # RAM requirement
    ram_node = evaluator.add_leaf(
        id="ram_requirement",
        desc="Verification shows RAM is at least 16GB",
        parent=specs_group,
        critical=True,
    )
    ram_claim = (
        f"The laptop RAM is at least 16GB. Extracted RAM spec: '{_safe_str(specs.ram)}'."
    )
    await evaluator.verify(
        claim=ram_claim,
        node=ram_node,
        sources=product_url,
        additional_instruction=(
            "Confirm total system memory is ≥ 16GB. If multiple configurations are shown, ensure the specific configuration implied by the answer is ≥ 16GB."
        ),
    )

    # Storage requirement
    storage_node = evaluator.add_leaf(
        id="storage_requirement",
        desc="Verification shows storage is at least 500GB SSD",
        parent=specs_group,
        critical=True,
    )
    storage_claim = (
        f"The laptop storage is at least 500GB SSD. Extracted storage spec: '{_safe_str(specs.storage)}'."
    )
    await evaluator.verify(
        claim=storage_claim,
        node=storage_node,
        sources=product_url,
        additional_instruction=(
            "Verify that the primary storage is solid-state (SSD/NVMe) and capacity ≥ 500GB. If the page shows 512GB or 1TB SSD, this should pass."
        ),
    )

    # Graphics requirement
    gpu_node = evaluator.add_leaf(
        id="graphics_requirement",
        desc="Verification shows dedicated NVIDIA or AMD GPU with at least 4GB VRAM",
        parent=specs_group,
        critical=True,
    )
    gpu_claim = (
        "The laptop has a dedicated NVIDIA or AMD discrete GPU with ≥ 4GB VRAM. "
        f"Extracted GPU spec: '{_safe_str(specs.gpu)}', VRAM: '{_safe_str(specs.vram)}'."
    )
    await evaluator.verify(
        claim=gpu_claim,
        node=gpu_node,
        sources=product_url,
        additional_instruction=(
            "Confirm the GPU is discrete (not integrated) and brand is NVIDIA or AMD. VRAM must be ≥ 4GB (e.g., 4GB, 6GB, 8GB GDDR6). "
            "If the page shows 'RTX 3050 6GB', this passes; integrated GPUs (e.g., Intel Iris Xe) should fail."
        ),
    )

    # OS requirement
    os_node = evaluator.add_leaf(
        id="os_requirement",
        desc="Verification shows operating system is Windows 11",
        parent=specs_group,
        critical=True,
    )
    os_claim = (
        f"The laptop's operating system is Windows 11. Extracted OS: '{_safe_str(specs.os)}'."
    )
    await evaluator.verify(
        claim=os_claim,
        node=os_node,
        sources=product_url,
        additional_instruction=(
            "Confirm the product page lists Windows 11 (Home or Pro acceptable). If only 'Windows 10' or 'No OS' is shown, this should fail."
        ),
    )

    # Display requirement
    display_node = evaluator.add_leaf(
        id="display_requirement",
        desc="Verification shows display resolution is at least 1920x1080 (Full HD)",
        parent=specs_group,
        critical=True,
    )
    display_claim = (
        f"The display resolution is at least 1920x1080 (Full HD). Extracted resolution: '{_safe_str(specs.display_resolution)}'."
    )
    await evaluator.verify(
        claim=display_claim,
        node=display_node,
        sources=product_url,
        additional_instruction=(
            "Confirm the native panel resolution is ≥ 1920x1080. Resolutions like 1920x1200, 2560x1440, or 3840x2160 also meet the requirement."
        ),
    )

    # Add ground truth info for clarity in summary
    evaluator.add_ground_truth({
        "uc_denver_requirements": UC_DENVER_REQUIREMENTS,
        "sale_period_target": "Black Friday / Cyber Monday 2025 (November 2025)"
    }, gt_type="requirements")


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
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
    Evaluate an agent's answer for the UC Denver gaming laptop BF/CM 2025 task.
    Returns a standardized summary with the verification tree and final score.
    """
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract laptop candidates and their details from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_laptops(),
        template_class=LaptopAnswerExtraction,
        extraction_name="laptop_candidates",
    )

    # Include a compact summary of the extracted first laptop for transparency
    first = _get_first_laptop(extraction)
    evaluator.add_custom_info(
        info={
            "product_name": first.product_name,
            "model_number": first.model_number,
            "product_url": first.product_url,
            "current_price": first.current_price,
            "gaming_marketing_text": first.gaming_marketing_text,
            "specs": (first.specs.dict() if first.specs else {})
        },
        info_type="extraction_summary",
        info_name="selected_laptop_summary"
    )

    # Build the verification tree and run verifications
    await build_verification_tree(evaluator, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()