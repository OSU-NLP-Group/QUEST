import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bestbuy_business_laptop_selection"
TASK_DESCRIPTION = """
I need to purchase a business laptop for remote professional work from Best Buy. Find a laptop that meets all of the following requirements:

Hardware Specifications:
- Processor: Intel Core i5 (13th generation or newer) OR Intel Core Ultra 5/7 OR AMD Ryzen 5 (7000 series or newer) OR AMD Ryzen 7
- RAM: Minimum 16GB
- Storage: Minimum 512GB SSD

Display Requirements:
- Screen size: Between 14 and 16 inches
- Display resolution: Minimum 1920×1080 (Full HD)

Software and Product Requirements:
- Operating System: Windows 11 Pro (not Home edition)
- Product Line: Must be from a recognized business laptop series (such as Dell Latitude, HP ProBook/EliteBook, Lenovo ThinkPad/ThinkBook, or equivalent business-grade series)

Connectivity:
- Must have USB-C port capability (including Thunderbolt variants)

Purchase Requirements:
- Must be currently available for purchase at Best Buy (online or for shipping)
- Price: Must not exceed $1,500 USD
- Must include manufacturer warranty coverage

Provide the laptop model name, key specifications, and the Best Buy product page URL.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LaptopExtraction(BaseModel):
    """
    Extract a single proposed laptop and its key details as stated by the answer.
    Strings are preferred to maximize robustness to formatting variations.
    """
    model_name: Optional[str] = None
    bestbuy_url: Optional[str] = None

    cpu: Optional[str] = None
    ram: Optional[str] = None
    storage: Optional[str] = None

    screen_size: Optional[str] = None
    resolution: Optional[str] = None

    os_edition: Optional[str] = None
    product_line: Optional[str] = None

    usb_c: Optional[str] = None

    price: Optional[str] = None
    availability: Optional[str] = None
    warranty: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_laptop() -> str:
    return """
    Extract the single laptop the answer proposes (if multiple are mentioned, extract the first one that appears to be the main recommendation). Return the following fields exactly as written in the answer:
    - model_name: The full laptop model name (e.g., "Dell Latitude 5440", "Lenovo ThinkPad T14 Gen 5").
    - bestbuy_url: The Best Buy product page URL for this exact laptop (must be a direct product page URL on bestbuy.com).
    - cpu: The processor description (e.g., "Intel Core i5-1340P", "Intel Core Ultra 7", "AMD Ryzen 7 7840U").
    - ram: The memory amount (e.g., "16GB", "32 GB").
    - storage: The storage description (e.g., "512GB SSD", "1 TB NVMe SSD").
    - screen_size: The screen size (e.g., "14-inch", "15.6\"", "16 inch").
    - resolution: The screen resolution (e.g., "1920 x 1080", "1920x1200", "2560 × 1600").
    - os_edition: The operating system edition (e.g., "Windows 11 Pro", "Windows 11 Home").
    - product_line: The series or family name (e.g., "Latitude", "EliteBook", "ProBook", "ThinkPad", "ThinkBook"). If unclear, write what the answer states.
    - usb_c: What the answer states about USB‑C/Thunderbolt ports (e.g., "2x USB-C (Thunderbolt 4)").
    - price: The price mentioned for the new unit (e.g., "$1,299.99"). If multiple, use the main "Your price"/current price for a new unit (not open-box).
    - availability: Any statement about availability/purchase (e.g., "In stock", "Available to ship").
    - warranty: Any mention of manufacturer warranty (e.g., "1-year manufacturer's warranty").

    If any field is not present in the answer, set it to null. Do not fabricate values. Extract only what the answer text claims.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _bestbuy_sources(extracted: LaptopExtraction) -> Optional[str | List[str]]:
    """Return Best Buy URL if present; otherwise None."""
    return extracted.bestbuy_url if extracted and extracted.bestbuy_url else None


def _safe_model(extracted: LaptopExtraction) -> str:
    return extracted.model_name.strip() if (extracted and extracted.model_name) else "the laptop on the referenced Best Buy page"


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, root_node, extracted: LaptopExtraction) -> None:
    """
    Build the rubric tree as specified and perform verifications against the Best Buy product page.
    All leaves are binary checks and critical where required by the rubric.
    """
    # Create a critical sequential node to represent the rubric root (as per JSON)
    business_root = evaluator.add_sequential(
        id="Business_Laptop_Selection",
        desc="Complete identification and verification of a business laptop meeting professional requirements",
        parent=root_node,
        critical=True
    )

    # Child 1: Laptop_Identification (parallel, critical)
    identification = evaluator.add_parallel(
        id="Laptop_Identification",
        desc="Identify a specific laptop model that meets all technical specifications and business requirements",
        parent=business_root,
        critical=True
    )

    # Hardware_Specifications (parallel, critical)
    hardware = evaluator.add_parallel(
        id="Hardware_Specifications",
        desc="Verify core hardware meets minimum business requirements",
        parent=identification,
        critical=True
    )

    # Processor requirement
    leaf_cpu = evaluator.add_leaf(
        id="Processor_Requirement",
        desc="Processor must be Intel Core i5 13th gen+ or Intel Core Ultra 5/7, or AMD Ryzen 5 7000 series+ or Ryzen 7",
        parent=hardware,
        critical=True
    )
    cpu_claim = (
        f"For {_safe_model(extracted)}, the processor shown on the Best Buy product page satisfies one of the following: "
        f"Intel Core i5 (13th gen or newer), Intel Core Ultra 5 or Ultra 7, AMD Ryzen 5 (7000 series or newer), or any AMD Ryzen 7."
    )
    await evaluator.verify(
        claim=cpu_claim,
        node=leaf_cpu,
        sources=_bestbuy_sources(extracted),
        additional_instruction=(
            "Use the Specifications/Overview text on the product page. Examples of valid CPUs: i5-1340P, i5-13500H, "
            "Core Ultra 5 125H, Core Ultra 7 155H, Ryzen 5 7640U/7640HS, Ryzen 7 7840U/7840HS, etc. "
            "12th‑gen i5 or Ryzen 5 5000‑series do NOT satisfy."
        )
    )

    # RAM requirement
    leaf_ram = evaluator.add_leaf(
        id="RAM_Requirement",
        desc="Must have minimum 16GB RAM",
        parent=hardware,
        critical=True
    )
    ram_claim = f"For {_safe_model(extracted)}, the product page indicates memory (RAM) is at least 16 GB."
    await evaluator.verify(
        claim=ram_claim,
        node=leaf_ram,
        sources=_bestbuy_sources(extracted),
        additional_instruction="Accept 16GB, 24GB, 32GB, etc. Phrases like '16 GB' or '16 Gigabytes' satisfy."
    )

    # Storage requirement
    leaf_storage = evaluator.add_leaf(
        id="Storage_Requirement",
        desc="Must have minimum 512GB SSD storage",
        parent=hardware,
        critical=True
    )
    storage_claim = f"For {_safe_model(extracted)}, the product page indicates storage is an SSD of at least 512 GB."
    await evaluator.verify(
        claim=storage_claim,
        node=leaf_storage,
        sources=_bestbuy_sources(extracted),
        additional_instruction="Accept 512GB SSD or larger (e.g., 1TB SSD). HDD alone does not satisfy."
    )

    # Display_Specifications (parallel, critical)
    display = evaluator.add_parallel(
        id="Display_Specifications",
        desc="Verify display meets requirements",
        parent=identification,
        critical=True
    )

    # Screen size range
    leaf_screen = evaluator.add_leaf(
        id="Screen_Size_Range",
        desc="Screen size must be between 14 and 16 inches",
        parent=display,
        critical=True
    )
    screen_claim = (
        f"For {_safe_model(extracted)}, the screen size shown on the Best Buy page is between 14.0 and 16.0 inches inclusive."
    )
    await evaluator.verify(
        claim=screen_claim,
        node=leaf_screen,
        sources=_bestbuy_sources(extracted),
        additional_instruction="Accept values like 14.0”, 14.5”, 15.6”, 16”. 13.x inches does NOT satisfy."
    )

    # Display resolution
    leaf_resolution = evaluator.add_leaf(
        id="Display_Resolution",
        desc="Display resolution must be minimum 1920×1080 (Full HD)",
        parent=display,
        critical=True
    )
    resolution_claim = (
        f"For {_safe_model(extracted)}, the display resolution shown on the Best Buy page is at least 1920×1080 or higher."
    )
    await evaluator.verify(
        claim=resolution_claim,
        node=leaf_resolution,
        sources=_bestbuy_sources(extracted),
        additional_instruction="1920x1080, 1920x1200, 2560x1600, 2880x1800, 3840x2160 all satisfy. 1366x768 does NOT."
    )

    # Software_Requirements (parallel, critical)
    software = evaluator.add_parallel(
        id="Software_Requirements",
        desc="Verify OS and business product line requirements",
        parent=identification,
        critical=True
    )

    # Operating System
    leaf_os = evaluator.add_leaf(
        id="Operating_System",
        desc="Must come with Windows 11 Pro (not Home edition)",
        parent=software,
        critical=True
    )
    os_claim = f"For {_safe_model(extracted)}, the Best Buy page indicates Windows 11 Pro (not Home) is preinstalled."
    await evaluator.verify(
        claim=os_claim,
        node=leaf_os,
        sources=_bestbuy_sources(extracted),
        additional_instruction="Accept 'Windows 11 Pro' or 'Windows 11 Pro for Business'. Reject Home edition or unspecified OS."
    )

    # Business product line
    leaf_business_line = evaluator.add_leaf(
        id="Business_Product_Line",
        desc="Must be from a recognized business laptop series (Dell Latitude, HP ProBook/EliteBook, Lenovo ThinkPad/ThinkBook, or equivalent)",
        parent=software,
        critical=True
    )
    business_claim = (
        f"The model {_safe_model(extracted)} belongs to a recognized business series (e.g., Dell Latitude, "
        f"HP ProBook/EliteBook, Lenovo ThinkPad/ThinkBook) or an equivalent enterprise/commercial line."
    )
    await evaluator.verify(
        claim=business_claim,
        node=leaf_business_line,
        sources=_bestbuy_sources(extracted),
        additional_instruction=(
            "Accept examples: Latitude, EliteBook, ProBook, ThinkPad, ThinkBook, Precision, ZBook. "
            "Reject consumer lines like Inspiron, Pavilion, IdeaPad (unless the page explicitly positions it as a business/commercial model)."
        )
    )

    # Connectivity_Requirements (parallel, critical)
    connectivity = evaluator.add_parallel(
        id="Connectivity_Requirements",
        desc="Verify required connectivity",
        parent=identification,
        critical=True
    )

    # USB-C capability
    leaf_usbc = evaluator.add_leaf(
        id="USB_C_Capability",
        desc="Must have USB-C port capability (including Thunderbolt variants)",
        parent=connectivity,
        critical=True
    )
    usbc_claim = f"For {_safe_model(extracted)}, the specifications indicate at least one USB-C / USB Type‑C / Thunderbolt 3/4/USB4 port."
    await evaluator.verify(
        claim=usbc_claim,
        node=leaf_usbc,
        sources=_bestbuy_sources(extracted),
        additional_instruction="Look for 'USB-C', 'USB Type‑C', 'Thunderbolt 3/4', or 'USB4' in the specs."
    )

    # Child 2: Purchase_Verification (parallel, critical)
    purchase = evaluator.add_parallel(
        id="Purchase_Verification",
        desc="Verify the identified laptop meets purchase and availability requirements",
        parent=business_root,
        critical=True
    )

    # Availability_and_Source (parallel, critical)
    availability_and_source = evaluator.add_parallel(
        id="Availability_and_Source",
        desc="Verify laptop is available from the specified retailer and provide the product page",
        parent=purchase,
        critical=True
    )

    # Product_URL
    leaf_url = evaluator.add_leaf(
        id="Product_URL",
        desc="Provide a valid Best Buy product page URL",
        parent=availability_and_source,
        critical=True
    )
    url_claim = "This webpage is a valid Best Buy product detail page for a laptop."
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=_bestbuy_sources(extracted),
        additional_instruction="The URL should be on bestbuy.com and represent a specific product page (not a search or category listing)."
    )

    # Best_Buy_Availability
    leaf_availability = evaluator.add_leaf(
        id="Best_Buy_Availability",
        desc="Laptop must be currently listed and available for purchase at Best Buy",
        parent=availability_and_source,
        critical=True
    )
    availability_claim = (
        "This Best Buy product page indicates the item is available for purchase online (e.g., an 'Add to Cart' option "
        "and/or 'Available for shipping' or similar). If 'Sold Out' or 'Unavailable', then it does not satisfy."
    )
    await evaluator.verify(
        claim=availability_claim,
        node=leaf_availability,
        sources=_bestbuy_sources(extracted),
        additional_instruction="Focus on current availability indicators on the page. 'Add to Cart' or similar denotes available."
    )

    # Cost_and_Warranty (parallel, critical)
    cost_warranty = evaluator.add_parallel(
        id="Cost_and_Warranty",
        desc="Verify pricing and warranty coverage",
        parent=purchase,
        critical=True
    )

    # Price_Constraint
    leaf_price = evaluator.add_leaf(
        id="Price_Constraint",
        desc="Total laptop price must not exceed $1,500 USD",
        parent=cost_warranty,
        critical=True
    )
    price_claim = "The current price for a new unit on this Best Buy product page does not exceed $1,500 USD."
    await evaluator.verify(
        claim=price_claim,
        node=leaf_price,
        sources=_bestbuy_sources(extracted),
        additional_instruction=(
            "Use the main 'Your price' or current new-unit price. Ignore open-box/used pricing. "
            "Sale price counts if it is the current purchase price."
        )
    )

    # Warranty_Coverage
    leaf_warranty = evaluator.add_leaf(
        id="Warranty_Coverage",
        desc="Must include manufacturer warranty coverage",
        parent=cost_warranty,
        critical=True
    )
    warranty_claim = (
        "This Best Buy product page indicates manufacturer warranty coverage (e.g., 'Manufacturer's Warranty – Parts/Labor')."
    )
    await evaluator.verify(
        claim=warranty_claim,
        node=leaf_warranty,
        sources=_bestbuy_sources(extracted),
        additional_instruction="Look for a 'Manufacturer's Warranty' section stating coverage (e.g., 1 year parts/labor)."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Best Buy business laptop selection task.
    """
    # Initialize evaluator with a sequential top-level strategy
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract proposed laptop details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_laptop(),
        template_class=LaptopExtraction,
        extraction_name="laptop_extraction"
    )

    # Add some custom info for debugging / visibility
    evaluator.add_custom_info(
        info={
            "model_name": extracted.model_name,
            "bestbuy_url": extracted.bestbuy_url,
            "cpu": extracted.cpu,
            "ram": extracted.ram,
            "storage": extracted.storage,
            "screen_size": extracted.screen_size,
            "resolution": extracted.resolution,
            "os_edition": extracted.os_edition,
            "product_line": extracted.product_line,
            "usb_c": extracted.usb_c,
            "price": extracted.price,
            "availability": extracted.availability,
            "warranty": extracted.warranty
        },
        info_type="extraction_summary",
        info_name="extracted_laptop_fields"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extracted)

    # Return structured evaluation result
    return evaluator.get_summary()