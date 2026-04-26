import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "tech_launches_2025_2026"
TASK_DESCRIPTION = """
A technology analyst is preparing a comprehensive report on major consumer technology product launches from early 2025 through Q1 2026. Identify four specific products that each meet ALL of the following criteria:

Product 1 — High-Performance Graphics Card:
- Built on NVIDIA's Blackwell GPU architecture
- Released on January 30, 2025
- Equipped with exactly 32 GB of GDDR7 memory
- Total graphics power (TGP) specification of 575 watts
- Launched at an MSRP starting at $1,999

Product 2 — Portable Bluetooth Keyboard Accessory:
- Announced at CES 2026 (held in early January 2026)
- Standalone Bluetooth keyboard that magnetically attaches to devices
- Supports simultaneous pairing with up to 3 devices
- Uses MagSafe or Qi2 magnetic mounting technology
- Manufacturer claims 50% increase in usable screen space

Product 3 — Dedicated Communication Mobile Device:
- Announced at CES 2026 on January 2, 2026
- Standalone mobile phone with integrated physical keyboard (not an accessory for other phones)
- Features a full physical QWERTY keyboard
- Retail price of $499
- Marketed as purpose-built for communication and productivity

Product 4 — Survival Horror Video Game:
- Released on February 27, 2026
- Part of the Resident Evil franchise published by Capcom
- Features an antagonist character named Zeno
- Available on at least 4 platforms including PlayStation 5, Xbox Series, PC, and Nintendo Switch 2

For each product, provide:
1. The official product name or model designation
2. The manufacturer or developer company name
3. At least one reference URL from an official source or major tech publication confirming the specifications
""".strip()


# -----------------------------------------------------------------------------
# Data Models (Extraction)
# -----------------------------------------------------------------------------
class ProductCommon(BaseModel):
    product_name: Optional[str] = None
    manufacturer_or_developer: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Product1Info(ProductCommon):
    architecture: Optional[str] = None
    release_date: Optional[str] = None
    memory: Optional[str] = None
    tgp: Optional[str] = None
    msrp: Optional[str] = None


class Product2Info(ProductCommon):
    announcement_event: Optional[str] = None
    announcement_date: Optional[str] = None
    device_type_description: Optional[str] = None
    pairing_capacity_text: Optional[str] = None
    mounting_tech: Optional[str] = None
    screen_space_claim_text: Optional[str] = None


class Product3Info(ProductCommon):
    announcement_event: Optional[str] = None
    announcement_date: Optional[str] = None
    device_type_description: Optional[str] = None
    keyboard_type_desc: Optional[str] = None
    price: Optional[str] = None
    marketing_positioning_text: Optional[str] = None


class Product4Info(ProductCommon):
    release_date: Optional[str] = None
    franchise_name: Optional[str] = None
    publisher: Optional[str] = None
    antagonist_name: Optional[str] = None
    platforms: List[str] = Field(default_factory=list)


class AllProductsExtraction(BaseModel):
    product1: Optional[Product1Info] = None
    product2: Optional[Product2Info] = None
    product3: Optional[Product3Info] = None
    product4: Optional[Product4Info] = None


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_products() -> str:
    return """
Extract exactly four products from the answer, one for each category below. Only extract information explicitly present in the answer. Do not infer or add new information. If a field is missing, set it to null. For any URLs, extract the actual URLs as they appear in the answer text (plain or in markdown links).

Return a JSON object with the following structure:

{
  "product1": {
    "product_name": string|null,
    "manufacturer_or_developer": string|null,
    "urls": string[],

    "architecture": string|null,
    "release_date": string|null,
    "memory": string|null,
    "tgp": string|null,
    "msrp": string|null
  },
  "product2": {
    "product_name": string|null,
    "manufacturer_or_developer": string|null,
    "urls": string[],

    "announcement_event": string|null,
    "announcement_date": string|null,
    "device_type_description": string|null,
    "pairing_capacity_text": string|null,
    "mounting_tech": string|null,
    "screen_space_claim_text": string|null
  },
  "product3": {
    "product_name": string|null,
    "manufacturer_or_developer": string|null,
    "urls": string[],

    "announcement_event": string|null,
    "announcement_date": string|null,
    "device_type_description": string|null,
    "keyboard_type_desc": string|null,
    "price": string|null,
    "marketing_positioning_text": string|null
  },
  "product4": {
    "product_name": string|null,
    "manufacturer_or_developer": string|null,
    "urls": string[],

    "release_date": string|null,
    "franchise_name": string|null,
    "publisher": string|null,
    "antagonist_name": string|null,
    "platforms": string[]
  }
}

CATEGORY DEFINITIONS AND MAPPING:
- product1: A high-performance graphics card. Extract GPU architecture, release date, memory capacity/type, TGP, and MSRP.
- product2: A portable standalone Bluetooth keyboard accessory. Extract CES 2026 announcement info, magnetic mounting tech (MagSafe or Qi2), pairing capacity, and claimed screen space benefit.
- product3: A dedicated mobile phone device with integrated physical keyboard (not an accessory). Extract CES 2026 announcement date (ideally Jan 2, 2026), keyboard type (full physical QWERTY), price, and marketing positioning.
- product4: A survival horror video game. Extract release date, franchise, publisher, antagonist name, and platforms list.

URL EXTRACTION RULES:
- Only include URLs explicitly present in the answer text. If none are present, return an empty list.
- Accept plain URLs or markdown links; always output the final URL strings.

Formatting notes:
- Keep dates and numeric information as strings exactly as presented in the answer (e.g., "January 30, 2025" or "Jan 30, 2025" or "2025-01-30").
- For platforms, extract them as a string array (e.g., ["PlayStation 5", "Xbox Series X|S", "PC", "Nintendo Switch 2"]).
    """.strip()


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _display_name(name: Optional[str]) -> str:
    return name if (name and name.strip()) else "the product"


def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls or [])


# -----------------------------------------------------------------------------
# Verification subroutines (one per product)
# -----------------------------------------------------------------------------
async def verify_product_1(evaluator: Evaluator, parent_node, p: Optional[Product1Info]) -> None:
    product_node = evaluator.add_parallel(
        id="Product_1_Graphics_Card",
        desc="Verify the high-performance graphics card meets all specified criteria",
        parent=parent_node,
        critical=False
    )
    urls = (p.urls if p else []) if p else []
    name = _display_name(p.product_name if p else None)

    # URL existence (critical gate)
    evaluator.add_custom_node(
        result=_has_any_url(urls),
        id="Product_1_URL_Reference",
        desc="At least one reference URL is provided from an official source or major tech publication",
        parent=product_node,
        critical=True
    )

    # Architecture
    n = evaluator.add_leaf(
        id="Product_1_Architecture",
        desc="The graphics card is built on NVIDIA's Blackwell architecture",
        parent=product_node,
        critical=True
    )
    claim = f"{name} is based on NVIDIA's Blackwell GPU architecture."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Verify the page states the product uses NVIDIA's Blackwell architecture (allow phrasing like 'Blackwell-based', GB20-family, or similar)."
    )

    # Release Date
    n = evaluator.add_leaf(
        id="Product_1_Release_Date",
        desc="The graphics card was released on January 30, 2025",
        parent=product_node,
        critical=True
    )
    claim = f"{name} was released on January 30, 2025."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Accept reasonable date formatting variants (e.g., 'Jan 30, 2025', '2025-01-30')."
    )

    # Memory
    n = evaluator.add_leaf(
        id="Product_1_Memory",
        desc="The graphics card has exactly 32 GB of GDDR7 memory",
        parent=product_node,
        critical=True
    )
    claim = f"{name} is equipped with exactly 32 GB of GDDR7 memory."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Look for phrases like '32GB GDDR7'. Do not accept other capacities or memory types."
    )

    # TGP
    n = evaluator.add_leaf(
        id="Product_1_TGP",
        desc="The graphics card has a total graphics power (TGP) specification of 575 watts",
        parent=product_node,
        critical=True
    )
    claim = f"{name} has a total graphics power (TGP) of 575 watts."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Allow synonyms like 'board power' if clearly equivalent; accept formatting '575W'."
    )

    # Price
    n = evaluator.add_leaf(
        id="Product_1_Price",
        desc="The graphics card launched at an MSRP starting at $1,999",
        parent=product_node,
        critical=True
    )
    claim = f"{name} launched at an MSRP starting at $1,999."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Accept variants like 'USD 1,999' or '$1,999 starting price'."
    )


async def verify_product_2(evaluator: Evaluator, parent_node, p: Optional[Product2Info]) -> None:
    product_node = evaluator.add_parallel(
        id="Product_2_Bluetooth_Keyboard",
        desc="Verify the portable Bluetooth keyboard accessory meets all specified criteria",
        parent=parent_node,
        critical=False
    )
    urls = (p.urls if p else []) if p else []
    name = _display_name(p.product_name if p else None)

    # URL existence (critical gate)
    evaluator.add_custom_node(
        result=_has_any_url(urls),
        id="Product_2_URL_Reference",
        desc="At least one reference URL is provided from an official source or major tech publication",
        parent=product_node,
        critical=True
    )

    # CES 2026 announcement
    n = evaluator.add_leaf(
        id="Product_2_CES_Announcement",
        desc="The keyboard was announced at CES 2026 held in early January 2026",
        parent=product_node,
        critical=True
    )
    claim = f"{name} was announced at CES 2026 (in early January 2026)."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Look for explicit mention of 'CES 2026' and/or a date in early January 2026 tied to the announcement."
    )

    # Device type: standalone Bluetooth keyboard that magnetically attaches
    n = evaluator.add_leaf(
        id="Product_2_Device_Type",
        desc="The product is a standalone Bluetooth keyboard that magnetically attaches to devices",
        parent=product_node,
        critical=True
    )
    claim = f"{name} is a standalone Bluetooth keyboard that magnetically attaches to devices."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Confirm it is a standalone Bluetooth keyboard (not an on-screen keyboard or a mere case) and that it magnetically attaches."
    )

    # Pairing capacity (up to 3 devices)
    n = evaluator.add_leaf(
        id="Product_2_Pairing_Capacity",
        desc="The keyboard supports simultaneous pairing with up to 3 devices",
        parent=product_node,
        critical=True
    )
    claim = f"{name} supports simultaneous pairing with up to 3 devices."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Accept wording like 'pair with three devices' or 'multi-device (3) pairing'."
    )

    # Mounting technology (MagSafe or Qi2)
    n = evaluator.add_leaf(
        id="Product_2_Mounting",
        desc="The keyboard uses MagSafe or Qi2 magnetic mounting technology",
        parent=product_node,
        critical=True
    )
    claim = f"{name} uses MagSafe or Qi2 magnetic mounting technology."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Accept phrasing like 'MagSafe-compatible', 'Qi2 magnetic attachment', 'Qi2 magnets'."
    )

    # Screen space claim (50%)
    n = evaluator.add_leaf(
        id="Product_2_Screen_Space",
        desc="The manufacturer claims the keyboard increases usable screen space by 50%",
        parent=product_node,
        critical=True
    )
    claim = f"The manufacturer claims that using {name} increases usable screen space by 50%."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Look specifically for a claim of '50% more screen space' or equivalent phrasing attributed to the manufacturer."
    )


async def verify_product_3(evaluator: Evaluator, parent_node, p: Optional[Product3Info]) -> None:
    product_node = evaluator.add_parallel(
        id="Product_3_Communication_Device",
        desc="Verify the dedicated communication mobile device meets all specified criteria",
        parent=parent_node,
        critical=False
    )
    urls = (p.urls if p else []) if p else []
    name = _display_name(p.product_name if p else None)

    # URL existence (critical gate)
    evaluator.add_custom_node(
        result=_has_any_url(urls),
        id="Product_3_URL_Reference",
        desc="At least one reference URL is provided from an official source or major tech publication",
        parent=product_node,
        critical=True
    )

    # CES 2026 on Jan 2, 2026
    n = evaluator.add_leaf(
        id="Product_3_CES_Announcement",
        desc="The device was announced at CES 2026 on January 2, 2026",
        parent=product_node,
        critical=True
    )
    claim = f"{name} was announced at CES 2026 on January 2, 2026."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Verify a CES 2026 announcement tied specifically to January 2, 2026 (accept reasonable date format variants)."
    )

    # Device type: standalone mobile phone with integrated physical keyboard (not accessory)
    n = evaluator.add_leaf(
        id="Product_3_Device_Type",
        desc="The product is a standalone mobile phone with integrated physical keyboard, not merely an accessory for other phones",
        parent=product_node,
        critical=True
    )
    claim = f"{name} is a standalone mobile phone with an integrated physical keyboard (not an accessory for other phones)."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Confirm it is a phone itself (not a case or add-on) and has a built-in hardware keyboard."
    )

    # Full physical QWERTY keyboard
    n = evaluator.add_leaf(
        id="Product_3_Keyboard",
        desc="The device features a full physical QWERTY keyboard",
        parent=product_node,
        critical=True
    )
    claim = f"{name} features a full physical QWERTY keyboard."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Look for explicit mention of 'QWERTY' and that it is a physical/hardware keyboard."
    )

    # Retail price $499
    n = evaluator.add_leaf(
        id="Product_3_Price",
        desc="The device has a retail price of $499",
        parent=product_node,
        critical=True
    )
    claim = f"{name} has a retail price of $499."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Accept currency formatting variants (e.g., 'USD 499', '$499.00')."
    )

    # Marketing: purpose-built for communication and productivity
    n = evaluator.add_leaf(
        id="Product_3_Marketing",
        desc="The device is marketed as purpose-built for communication and productivity",
        parent=product_node,
        critical=True
    )
    claim = f"{name} is marketed as purpose-built for communication and productivity."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Look for marketing statements emphasizing communication-centric and productivity-focused use cases."
    )


async def verify_product_4(evaluator: Evaluator, parent_node, p: Optional[Product4Info]) -> None:
    product_node = evaluator.add_parallel(
        id="Product_4_Video_Game",
        desc="Verify the survival horror video game meets all specified criteria",
        parent=parent_node,
        critical=False
    )
    urls = (p.urls if p else []) if p else []
    name = _display_name(p.product_name if p else None)

    # URL existence (critical gate)
    evaluator.add_custom_node(
        result=_has_any_url(urls),
        id="Product_4_URL_Reference",
        desc="At least one reference URL is provided from an official source or major tech publication",
        parent=product_node,
        critical=True
    )

    # Release date
    n = evaluator.add_leaf(
        id="Product_4_Release_Date",
        desc="The game was released on February 27, 2026",
        parent=product_node,
        critical=True
    )
    claim = f"{name} was released on February 27, 2026."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Accept reasonable date formats; verify the official release date."
    )

    # Franchise + publisher
    n = evaluator.add_leaf(
        id="Product_4_Franchise",
        desc="The game is part of the Resident Evil franchise published by Capcom",
        parent=product_node,
        critical=True
    )
    claim = f"{name} is a part of the Resident Evil franchise and is published by Capcom."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Confirm both: (1) it belongs to the Resident Evil franchise, and (2) Capcom is the publisher."
    )

    # Antagonist Zeno
    n = evaluator.add_leaf(
        id="Product_4_Character",
        desc="The game features an antagonist character named Zeno",
        parent=product_node,
        critical=True
    )
    claim = f"{name} features an antagonist named Zeno."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction="Look for character details confirming an antagonist named 'Zeno'."
    )

    # Platforms including PS5, Xbox Series, PC, and Nintendo Switch 2 (at least those four)
    n = evaluator.add_leaf(
        id="Product_4_Platforms",
        desc="The game is available on at least 4 platforms including PlayStation 5, Xbox Series, PC, and Nintendo Switch 2",
        parent=product_node,
        critical=True
    )
    claim = f"{name} is available on PlayStation 5, Xbox Series, PC, and Nintendo Switch 2."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=urls,
        additional_instruction=(
            "Confirm availability on all four named platforms. "
            "Accept reasonable naming variants: 'PS5' for PlayStation 5; 'Xbox Series X|S' for Xbox Series; "
            "'PC' may appear as 'Windows', 'Steam (PC)', or equivalent; "
            "For Nintendo Switch 2, if the source clearly refers to the next-gen 'Switch 2' or 'Switch successor' by that name, accept it."
        )
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate the agent's answer for identifying four products matching specified criteria (early 2025 through Q1 2026).
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Products evaluated independently
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_products(),
        template_class=AllProductsExtraction,
        extraction_name="products_extraction",
    )

    # Build product verification subtrees
    # Root should be non-critical to allow partial credit across products
    # Each product's internal leaves for constraints are critical (all must pass for that product)
    await verify_product_1(evaluator, root, extracted.product1 if extracted else None)
    await verify_product_2(evaluator, root, extracted.product2 if extracted else None)
    await verify_product_3(evaluator, root, extracted.product3 if extracted else None)
    await verify_product_4(evaluator, root, extracted.product4 if extracted else None)

    # Return standard summary
    return evaluator.get_summary()