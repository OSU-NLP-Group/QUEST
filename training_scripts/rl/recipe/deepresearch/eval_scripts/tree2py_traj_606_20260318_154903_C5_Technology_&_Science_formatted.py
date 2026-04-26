import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tech_products_2025_2026"
TASK_DESCRIPTION = """
Identify four technology products from 2025-2026 that match the following criteria:

Product 1: A budget iPhone model that became generally available on March 11, 2026, featuring:
- Apple's A19 chip with a 4-core GPU (not the 5-core version)
- Apple's C1X 5G modem
- 256GB starting storage capacity
- $599 USD starting price

Product 2: A PlayStation 5 accessory collection that launched on March 12, 2026, featuring:
- Exactly three color variants named Techno Red, Remix Green, and Rhythm Blue
- Both DualSense wireless controllers and PS5 console covers

Product 3: A flagship smartphone series that was announced on February 25, 2026 at Samsung Unpacked, featuring:
- The Ultra model using Snapdragon 8 Elite Gen 5 processor globally (all regions)
- The base and Plus models using Snapdragon 8 Elite Gen 5 in the US and select regions, but Exynos 2600 in Europe, South Korea, and Malaysia

Product 4: An NFL video game that released on August 14, 2025, featuring:
- Platform support for PlayStation 5, Xbox Series X|S, Nintendo Switch 2, and PC

For each product, provide the product name and a reference URL that confirms the specifications.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProductItem(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class TechProductsExtraction(BaseModel):
    product1: Optional[ProductItem] = None  # Budget iPhone
    product2: Optional[ProductItem] = None  # PS5 accessory collection
    product3: Optional[ProductItem] = None  # Samsung flagship smartphone series
    product4: Optional[ProductItem] = None  # NFL video game


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_products() -> str:
    return """
    Extract exactly four products mentioned in the answer. For each product, extract:
    - name: The product name exactly as written in the answer (string).
    - reference_urls: An array of URL(s) that the answer explicitly cites as evidence or reference for that product.
    
    The four products, in order, are:
    1) product1: the budget iPhone model.
    2) product2: the PS5 accessory collection.
    3) product3: the Samsung flagship smartphone series.
    4) product4: the NFL video game.

    IMPORTANT:
    - Only include URLs that are explicitly present in the answer text (plain or markdown links). Do not invent URLs.
    - If multiple URLs are cited for a product, include them all in reference_urls.
    - If no URL is provided in the answer for a product, set reference_urls to an empty list and name to the provided product name if present, otherwise null.
    - Do not attempt to infer or validate facts here; only extract what the answer states.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def safe_sources(item: Optional[ProductItem]) -> List[str]:
    if item and item.reference_urls:
        # De-duplicate while preserving order
        seen = set()
        uniq = []
        for u in item.reference_urls:
            if isinstance(u, str) and u.strip() and u.strip() not in seen:
                uniq.append(u.strip())
                seen.add(u.strip())
        return uniq
    return []


def product_display_name(item: Optional[ProductItem], fallback: str) -> str:
    if item and item.name and item.name.strip():
        return item.name.strip()
    return fallback


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_product_1_budget_iphone(evaluator: Evaluator, parent, item: Optional[ProductItem]) -> None:
    node = evaluator.add_parallel(
        id="product_1_budget_iphone",
        desc="Identify the budget iPhone model that became available in March 2026",
        parent=parent,
        critical=False
    )

    sources = safe_sources(item)
    name = product_display_name(item, "the budget iPhone model")

    # Reference presence (critical)
    evaluator.add_custom_node(
        result=(item is not None and bool(sources)),
        id="iphone_reference",
        desc="Provide a reference URL supporting the iPhone identification (presence check)",
        parent=node,
        critical=True
    )

    # Availability date (critical)
    leaf_avail = evaluator.add_leaf(
        id="iphone_availability",
        desc="The iPhone model became generally available on March 11, 2026",
        parent=node,
        critical=True
    )
    claim_avail = f"{name} became generally available on March 11, 2026."
    await evaluator.verify(
        claim=claim_avail,
        node=leaf_avail,
        sources=sources,
        additional_instruction="Confirm the general availability date is March 11, 2026. Accept reasonable phrasing variants like 'available starting March 11, 2026'."
    )

    # Technical specs (parallel subgroup)
    specs_group = evaluator.add_parallel(
        id="iphone_technical_specs",
        desc="Verify the iPhone model's technical specifications match the constraints",
        parent=node,
        critical=False
    )

    # Chip + GPU cores (critical)
    leaf_chip = evaluator.add_leaf(
        id="chip_gpu_cores",
        desc="Uses A19 chip with 4-core GPU (distinguishing it from the 5-core GPU in standard iPhone 17)",
        parent=specs_group,
        critical=True
    )
    claim_chip = f"{name} uses Apple's A19 chip with a 4-core GPU, not the 5-core GPU variant."
    await evaluator.verify(
        claim=claim_chip,
        node=leaf_chip,
        sources=sources,
        additional_instruction="Look for the specific GPU core count (4-core) associated with A19 for this model and that it is NOT the 5-core variant."
    )

    # Modem type (critical)
    leaf_modem = evaluator.add_leaf(
        id="modem_type",
        desc="Features Apple's C1X 5G modem",
        parent=specs_group,
        critical=True
    )
    claim_modem = f"{name} features Apple's C1X 5G modem."
    await evaluator.verify(
        claim=claim_modem,
        node=leaf_modem,
        sources=sources,
        additional_instruction="Confirm the modem branding 'C1X' is explicitly stated for this model."
    )

    # Storage + Price (critical)
    leaf_storage_price = evaluator.add_leaf(
        id="storage_price",
        desc="Starting storage is 256GB at $599 USD",
        parent=specs_group,
        critical=True
    )
    claim_storage_price = f"{name} has a starting storage of 256GB and a starting price of $599 USD."
    await evaluator.verify(
        claim=claim_storage_price,
        node=leaf_storage_price,
        sources=sources,
        additional_instruction="Check official spec sheets or announcements for the base storage (256GB) and the starting price ($599)."
    )


async def build_product_2_ps5_collection(evaluator: Evaluator, parent, item: Optional[ProductItem]) -> None:
    node = evaluator.add_parallel(
        id="product_2_ps5_collection",
        desc="Identify the PS5 accessory collection that launched in March 2026",
        parent=parent,
        critical=False
    )

    sources = safe_sources(item)
    name = product_display_name(item, "the PS5 accessory collection")

    # Reference presence (critical)
    evaluator.add_custom_node(
        result=(item is not None and bool(sources)),
        id="collection_reference",
        desc="Provide a reference URL supporting the collection identification (presence check)",
        parent=node,
        critical=True
    )

    # Details (parallel subgroup)
    details = evaluator.add_parallel(
        id="collection_details",
        desc="Verify the collection's details match the constraints",
        parent=node,
        critical=False
    )

    # Launch date (critical)
    leaf_launch = evaluator.add_leaf(
        id="launch_date",
        desc="Launched on March 12, 2026",
        parent=details,
        critical=True
    )
    claim_launch = f"{name} launched on March 12, 2026."
    await evaluator.verify(
        claim=claim_launch,
        node=leaf_launch,
        sources=sources,
        additional_instruction="Confirm the public launch/release date is March 12, 2026."
    )

    # Color options (critical)
    leaf_colors = evaluator.add_leaf(
        id="color_options",
        desc="Includes exactly three colors: Techno Red, Remix Green, and Rhythm Blue",
        parent=details,
        critical=True
    )
    claim_colors = f"{name} includes exactly three color variants named 'Techno Red', 'Remix Green', and 'Rhythm Blue' (no more, no fewer)."
    await evaluator.verify(
        claim=claim_colors,
        node=leaf_colors,
        sources=sources,
        additional_instruction="Verify there are exactly three color variants with those exact names. Allow minor punctuation/casing variations."
    )

    # Product categories (critical)
    leaf_categories = evaluator.add_leaf(
        id="product_categories",
        desc="Includes both DualSense wireless controllers and console covers",
        parent=details,
        critical=True
    )
    claim_categories = f"{name} includes both DualSense wireless controllers and PS5 console covers."
    await evaluator.verify(
        claim=claim_categories,
        node=leaf_categories,
        sources=sources,
        additional_instruction="Confirm both categories are present in this collection: controllers and console covers."
    )


async def build_product_3_flagship_series(evaluator: Evaluator, parent, item: Optional[ProductItem]) -> None:
    node = evaluator.add_parallel(
        id="product_3_flagship_series",
        desc="Identify the flagship smartphone series announced at Samsung Unpacked in February 2026",
        parent=parent,
        critical=False
    )

    sources = safe_sources(item)
    name = product_display_name(item, "the flagship Samsung smartphone series")

    # Reference presence (critical)
    evaluator.add_custom_node(
        result=(item is not None and bool(sources)),
        id="series_reference",
        desc="Provide a reference URL supporting the series identification (presence check)",
        parent=node,
        critical=True
    )

    # Announcement event (critical)
    leaf_announce = evaluator.add_leaf(
        id="announcement_event",
        desc="Announced on February 25, 2026 at Samsung Unpacked",
        parent=node,
        critical=True
    )
    claim_announce = f"{name} was announced on February 25, 2026 at Samsung Unpacked."
    await evaluator.verify(
        claim=claim_announce,
        node=leaf_announce,
        sources=sources,
        additional_instruction="Verify the exact date (Feb 25, 2026) and that the venue is Samsung Unpacked."
    )

    # Processor specifications (parallel subgroup)
    proc = evaluator.add_parallel(
        id="processor_specifications",
        desc="Verify processor specifications by model and region",
        parent=node,
        critical=False
    )

    # Ultra processor globally (critical)
    leaf_ultra = evaluator.add_leaf(
        id="ultra_processor",
        desc="Ultra model uses Snapdragon 8 Elite Gen 5 globally in all regions",
        parent=proc,
        critical=True
    )
    claim_ultra = f"The Ultra model in {name} uses Snapdragon 8 Elite Gen 5 in all regions globally."
    await evaluator.verify(
        claim=claim_ultra,
        node=leaf_ultra,
        sources=sources,
        additional_instruction="Confirm that the Ultra model is Snapdragon-only globally. The page should not indicate any Exynos or regional variance for Ultra."
    )

    # Base/Plus regional split (critical)
    leaf_base_plus = evaluator.add_leaf(
        id="base_models_processor",
        desc="Base and Plus models use Snapdragon 8 Elite Gen 5 in US/select regions, Exynos 2600 in Europe/South Korea/Malaysia",
        parent=proc,
        critical=True
    )
    claim_base_plus = (
        f"The base and Plus models in {name} use Snapdragon 8 Elite Gen 5 in the US and select regions, "
        f"but use Exynos 2600 in Europe, South Korea, and Malaysia."
    )
    await evaluator.verify(
        claim=claim_base_plus,
        node=leaf_base_plus,
        sources=sources,
        additional_instruction="Verify both parts: Snapdragon 8 Elite Gen 5 in the US/select regions; Exynos 2600 in Europe, South Korea, and Malaysia. The evidence should clearly indicate the split."
    )


async def build_product_4_nfl_game(evaluator: Evaluator, parent, item: Optional[ProductItem]) -> None:
    node = evaluator.add_parallel(
        id="product_4_nfl_game",
        desc="Identify the NFL video game that released on August 14, 2025 with Nintendo Switch 2 support",
        parent=parent,
        critical=False
    )

    sources = safe_sources(item)
    name = product_display_name(item, "the NFL video game")

    # Reference presence (critical)
    evaluator.add_custom_node(
        result=(item is not None and bool(sources)),
        id="game_reference",
        desc="Provide a reference URL supporting the game identification (presence check)",
        parent=node,
        critical=True
    )

    # Release details (parallel subgroup)
    details = evaluator.add_parallel(
        id="game_release_details",
        desc="Verify the game's release details",
        parent=node,
        critical=False
    )

    # Release date (critical)
    leaf_release = evaluator.add_leaf(
        id="release_date",
        desc="Released on August 14, 2025",
        parent=details,
        critical=True
    )
    claim_release = f"{name} released on August 14, 2025."
    await evaluator.verify(
        claim=claim_release,
        node=leaf_release,
        sources=sources,
        additional_instruction="Confirm that the release date is August 14, 2025."
    )

    # Platform support (critical)
    leaf_platforms = evaluator.add_leaf(
        id="platform_support",
        desc="Available on PlayStation 5, Xbox Series X|S, Nintendo Switch 2, and PC",
        parent=details,
        critical=True
    )
    claim_platforms = f"{name} is available on PlayStation 5, Xbox Series X|S, Nintendo Switch 2, and PC."
    await evaluator.verify(
        claim=claim_platforms,
        node=leaf_platforms,
        sources=sources,
        additional_instruction="Verify all four platforms are officially listed for this title. Allow minor naming variants (e.g., PC/Windows)."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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

    # Record the constraints in the output for reference
    evaluator.add_ground_truth({
        "product_1_requirements": {
            "availability": "March 11, 2026",
            "chip_gpu": "A19 with 4-core GPU (not 5-core)",
            "modem": "Apple C1X 5G",
            "base_storage": "256GB",
            "starting_price": "$599 USD",
        },
        "product_2_requirements": {
            "launch_date": "March 12, 2026",
            "colors": ["Techno Red", "Remix Green", "Rhythm Blue"],
            "categories": ["DualSense wireless controllers", "PS5 console covers"],
        },
        "product_3_requirements": {
            "announcement_event": "Samsung Unpacked, Feb 25, 2026",
            "ultra": "Snapdragon 8 Elite Gen 5 globally (all regions)",
            "base_plus": "Snapdragon 8 Elite Gen 5 in US/select; Exynos 2600 in Europe/South Korea/Malaysia",
        },
        "product_4_requirements": {
            "release_date": "August 14, 2025",
            "platforms": ["PlayStation 5", "Xbox Series X|S", "Nintendo Switch 2", "PC"],
        }
    })

    # Extract structured product entries
    extracted = await evaluator.extract(
        prompt=prompt_extract_products(),
        template_class=TechProductsExtraction,
        extraction_name="products_extraction",
    )

    # Build verification subtrees
    await build_product_1_budget_iphone(evaluator, root, extracted.product1)
    await build_product_2_ps5_collection(evaluator, root, extracted.product2)
    await build_product_3_flagship_series(evaluator, root, extracted.product3)
    await build_product_4_nfl_game(evaluator, root, extracted.product4)

    # Return standard summary
    return evaluator.get_summary()