import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "apple_early_2026_n1_wifi7_bt6"
TASK_DESCRIPTION = """
Identify 4 Apple products (Macs, iPads, or iPhones) that were officially announced between January 1, 2026 and March 31, 2026, and that feature Apple's N1 networking chip with Wi-Fi 7 and Bluetooth 6 support. For each product, provide: (1) the official product name, (2) the official announcement date, (3) the chip model (M5, M4, or A19 series), (4) confirmation of N1 networking chip with Wi-Fi 7 and Bluetooth 6 support, (5) the base storage configuration, (6) the starting U.S. price, and (7) a direct link to the official Apple announcement page or product page on apple.com.
"""

DATE_RANGE_START = "2026-01-01"
DATE_RANGE_END = "2026-03-31"


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class ProductItem(BaseModel):
    # Prefer strings for robustness; the verifier will read from the official page
    name: Optional[str] = None
    category: Optional[str] = None  # "Mac", "iPad", or "iPhone" if available
    announcement_date: Optional[str] = None  # any human-readable date string
    chip_model: Optional[str] = None  # e.g., "M5", "M5 Pro", "M4", "A19 Pro", "A19 Bionic"
    networking_n1: Optional[str] = None  # "yes"/"no"/"unknown" (not strictly required for verification)
    networking_wifi7: Optional[str] = None
    networking_bluetooth6: Optional[str] = None
    networking_thread: Optional[str] = None  # optional
    base_storage: Optional[str] = None  # e.g., "128GB", "256GB"
    starting_price_usd: Optional[str] = None  # e.g., "$999"
    apple_url: Optional[str] = None  # must be an official Apple URL (apple.com domain)


class ProductsExtraction(BaseModel):
    products: List[ProductItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_products() -> str:
    return """
Extract up to the first 4 Apple products mentioned in the answer that are Macs, iPads, or iPhones. For each, extract EXACTLY these fields:

- name: Official product name as written by Apple (string).
- category: One of "Mac", "iPad", or "iPhone" (string). Infer from the product name if needed.
- announcement_date: The date the product was officially announced (string, keep the original format in the answer, e.g., "March 15, 2026").
- chip_model: The Apple silicon model claimed (e.g., "M5", "M5 Pro", "M4", "A19", "A19 Pro", "A19 Bionic") (string).
- networking_n1: Whether the answer explicitly claims Apple's N1 networking chip (string like "yes"/"no" or the phrase used; return null if absent).
- networking_wifi7: Whether the answer explicitly claims Wi‑Fi 7 support (string or phrase; return null if absent).
- networking_bluetooth6: Whether the answer explicitly claims Bluetooth 6 support (string or phrase; return null if absent).
- networking_thread: Whether the answer explicitly claims Thread support (string or phrase; can be null if absent).
- base_storage: The base storage configuration (e.g., "128GB", "256GB") (string).
- starting_price_usd: The starting U.S. price as stated (e.g., "$999") (string).
- apple_url: A direct URL to an official Apple page about this product (Newsroom announcement or product page on an apple.com domain). 
             Extract only URLs explicitly present in the answer and on apple.com (including newsroom.apple.com and localized subdomains). If none, return null.

Rules:
- Only include Macs, iPads, or iPhones (exclude Watch, AirPods, accessories, services).
- Do not fabricate URLs; extract them exactly as provided. If a URL is missing protocol, prepend http:// per URL extraction rules.
- Return a JSON object with a single key "products" mapping to an array of up to 4 product objects, each with the exact fields listed above.
- If any field is missing for a product, set it to null.
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def infer_category(name: Optional[str], provided_category: Optional[str]) -> Optional[str]:
    if provided_category:
        cat = provided_category.strip().lower()
        if "mac" in cat:
            return "Mac"
        if "ipad" in cat:
            return "iPad"
        if "iphone" in cat:
            return "iPhone"
    if name:
        lower = name.lower()
        if "ipad" in lower:
            return "iPad"
        if "iphone" in lower:
            return "iPhone"
        if any(k in lower for k in ["macbook", "imac", "mac mini", "mac studio", "mac pro", "macbook pro", "macbook air"]):
            return "Mac"
        if lower.startswith("mac "):  # generic "Mac ..."
            return "Mac"
    return None


def normalize_nonempty(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = s.strip()
    return t if t else None


# -----------------------------------------------------------------------------
# Verification per product
# -----------------------------------------------------------------------------
async def verify_product(evaluator: Evaluator, parent_node, product: ProductItem, index_one_based: int) -> None:
    """
    Build and execute verification nodes for a single product.
    JSON adjustments:
    - The original JSON marks the 'networking' group as CRITICAL while having a NON-CRITICAL 'thread' child.
      The framework prohibits non-critical children under a critical parent. To respect both intentions:
        * We keep networking (N1/Wi‑Fi 7/Bluetooth 6) as a CRITICAL subgroup.
        * We place 'thread' as a separate NON-CRITICAL leaf under the product node.
    """

    pid = index_one_based  # 1..4
    pname = normalize_nonempty(product.name) or f"Product #{pid}"
    purl = normalize_nonempty(product.apple_url)
    pcat = infer_category(product.name, product.category)
    pchip = normalize_nonempty(product.chip_model)

    # Product container (NON-CRITICAL to allow partial credit across products)
    product_node = evaluator.add_parallel(
        id=f"product_{pid}",
        desc=f"{['First','Second','Third','Fourth'][pid-1]} qualifying product with required specifications",
        parent=parent_node,
        critical=False
    )

    # 1) URL presence and official page verification (CRITICAL at product level)
    url_leaf = evaluator.add_leaf(
        id=f"product_{pid}_url",
        desc="Direct link to official Apple announcement page or product page on apple.com or Apple Newsroom",
        parent=product_node,
        critical=True,
        status="initialized"
    )
    if purl and ("apple.com" in purl):
        await evaluator.verify(
            claim=f"This webpage is an official Apple page (apple.com domain) and is the announcement or product page for '{pname}'.",
            node=url_leaf,
            sources=purl,
            additional_instruction="Accept www.apple.com, apple.com, newsroom.apple.com, and localized subdomains (e.g., www.apple.com/xx-yy/). The page should clearly correspond to the named product."
        )
    else:
        # No valid Apple URL provided -> fail this critical leaf
        url_leaf.score = 0.0
        url_leaf.status = "failed"

    # 2) Identification group: name (URL-verified) and date-in-range (logic check)
    identification_node = evaluator.add_parallel(
        id=f"product_{pid}_identification",
        desc="Product name and announcement date verification",
        parent=product_node,
        critical=True
    )

    # 2.1 Name verification against the Apple page (CRITICAL)
    name_leaf = evaluator.add_leaf(
        id=f"product_{pid}_name",
        desc="Official product name as announced by Apple",
        parent=identification_node,
        critical=True,
        status="initialized"
    )
    if pname and purl:
        await evaluator.verify(
            claim=f"This Apple page is about a product officially named '{pname}'.",
            node=name_leaf,
            sources=purl,
            additional_instruction="Allow minor punctuation/casing variants. The page title or prominent headings should clearly match or be an equivalent of the stated product name."
        )
    else:
        name_leaf.score = 0.0
        name_leaf.status = "failed"

    # 2.2 Announcement date within Jan 1 – Mar 31, 2026 (CRITICAL)
    date_leaf = evaluator.add_leaf(
        id=f"product_{pid}_date",
        desc="Announcement date between January 1, 2026 and March 31, 2026",
        parent=identification_node,
        critical=True,
        status="initialized"
    )
    if normalize_nonempty(product.announcement_date):
        await evaluator.verify(
            claim=f"The date '{product.announcement_date}' falls between {DATE_RANGE_START} and {DATE_RANGE_END}.",
            node=date_leaf,
            additional_instruction="Interpret natural-language dates (e.g., 'March 15, 2026') and inclusive range boundaries. Output Correct only if the date is within or equal to the range bounds."
        )
    else:
        date_leaf.score = 0.0
        date_leaf.status = "failed"

    # 3) Chip model (CRITICAL): verify page confirms the claimed chip; also check suitability w.r.t category
    chip_group = evaluator.add_parallel(
        id=f"product_{pid}_chip",
        desc="Apple silicon chip specifications",
        parent=product_node,
        critical=True
    )

    chip_leaf = evaluator.add_leaf(
        id=f"product_{pid}_chip_model",
        desc="Chip model is latest-generation: M5 for Mac, M4 for iPad, or A19 series for iPhone",
        parent=chip_group,
        critical=True,
        status="initialized"
    )
    if pchip and purl:
        add_ins = (
            "Verify the page clearly states the device uses the claimed chip model. "
            "Also ensure the model is appropriate for its category: "
            "Mac devices should list an M5-family chip (e.g., M5, M5 Pro/Max/Ultra), "
            "iPad devices should list an M4-family chip (e.g., M4, M4 Pro), "
            "iPhone devices should list an A19-family chip (e.g., A19, A19 Pro, A19 Bionic). "
            "Treat suffixes like Pro/Max/Ultra/Bionic as within the same family."
        )
        if pcat:
            add_ins += f" The inferred/declared category is '{pcat}'."
        await evaluator.verify(
            claim=f"This Apple page confirms that '{pname}' uses the chip model '{pchip}'.",
            node=chip_leaf,
            sources=purl,
            additional_instruction=add_ins
        )
    else:
        chip_leaf.score = 0.0
        chip_leaf.status = "failed"

    # 4) Networking capabilities
    # 4.a Networking core (CRITICAL): N1 + Wi‑Fi 7 + Bluetooth 6 must be confirmed on the Apple page
    networking_group = evaluator.add_parallel(
        id=f"product_{pid}_networking",
        desc="N1 networking chip with complete wireless capabilities",
        parent=product_node,
        critical=True
    )

    n1_leaf = evaluator.add_leaf(
        id=f"product_{pid}_n1_chip",
        desc="Apple N1 networking chip confirmed",
        parent=networking_group,
        critical=True,
        status="initialized"
    )
    if purl:
        await evaluator.verify(
            claim=f"This Apple page confirms that '{pname}' includes Apple's N1 networking chip.",
            node=n1_leaf,
            sources=purl,
            additional_instruction="Look for explicit mention of 'N1' as Apple's networking chip (allow minor naming variants like 'N‑1'). Reject if only generic 'networking' is mentioned without 'N1'."
        )
    else:
        n1_leaf.score = 0.0
        n1_leaf.status = "failed"

    wifi7_leaf = evaluator.add_leaf(
        id=f"product_{pid}_wifi7",
        desc="Wi-Fi 7 (802.11be) support confirmed",
        parent=networking_group,
        critical=True,
        status="initialized"
    )
    if purl:
        await evaluator.verify(
            claim=f"This Apple page confirms that '{pname}' supports Wi‑Fi 7 (802.11be).",
            node=wifi7_leaf,
            sources=purl,
            additional_instruction="Accept 'Wi‑Fi 7', 'WiFi 7', or explicit '802.11be'. Reject if only Wi‑Fi 6/6E or earlier is mentioned."
        )
    else:
        wifi7_leaf.score = 0.0
        wifi7_leaf.status = "failed"

    bt6_leaf = evaluator.add_leaf(
        id=f"product_{pid}_bluetooth6",
        desc="Bluetooth 6 support confirmed",
        parent=networking_group,
        critical=True,
        status="initialized"
    )
    if purl:
        await evaluator.verify(
            claim=f"This Apple page confirms that '{pname}' supports Bluetooth 6.",
            node=bt6_leaf,
            sources=purl,
            additional_instruction="Accept 'Bluetooth 6' or 'Bluetooth 6.0'. Reject if only Bluetooth 5.x or earlier is mentioned."
        )
    else:
        bt6_leaf.score = 0.0
        bt6_leaf.status = "failed"

    # 4.b Thread (NON-CRITICAL), kept outside the critical networking group to satisfy framework constraints
    thread_leaf = evaluator.add_leaf(
        id=f"product_{pid}_thread",
        desc="Thread support confirmed",
        parent=product_node,
        critical=False,
        status="initialized"
    )
    if purl:
        await evaluator.verify(
            claim=f"This Apple page confirms that '{pname}' supports Thread.",
            node=thread_leaf,
            sources=purl,
            additional_instruction="Look for 'Thread' support (often in Home/Accessory connectivity contexts). Consider wording variations; fail if no indication of Thread."
        )
    else:
        # Non-critical; if URL missing, simply fail this leaf
        thread_leaf.score = 0.0
        thread_leaf.status = "failed"

    # 5) Base storage (CRITICAL)
    storage_leaf = evaluator.add_leaf(
        id=f"product_{pid}_storage",
        desc="Base storage configuration publicly disclosed",
        parent=product_node,
        critical=True,
        status="initialized"
    )
    if normalize_nonempty(product.base_storage) and purl:
        await evaluator.verify(
            claim=f"This Apple page shows the base storage (entry configuration) for '{pname}' is '{product.base_storage}'.",
            node=storage_leaf,
            sources=purl,
            additional_instruction="Match the lowest tier/base model storage capacity exactly (e.g., 128GB, 256GB). Do not confuse with higher-tier capacities."
        )
    else:
        storage_leaf.score = 0.0
        storage_leaf.status = "failed"

    # 6) Starting U.S. price (CRITICAL)
    price_leaf = evaluator.add_leaf(
        id=f"product_{pid}_price",
        desc="Starting U.S. price in dollars publicly disclosed",
        parent=product_node,
        critical=True,
        status="initialized"
    )
    if normalize_nonempty(product.starting_price_usd) and purl:
        await evaluator.verify(
            claim=f"This Apple page shows the starting U.S. price for '{pname}' is '{product.starting_price_usd}'.",
            node=price_leaf,
            sources=purl,
            additional_instruction="Look for 'From $X' in USD. Ignore trade-in or carrier promos. If page is localized, still verify equivalence to the claimed USD price if clearly stated."
        )
    else:
        price_leaf.score = 0.0
        price_leaf.status = "failed"


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    Entry point for evaluating the Apple early-2026 N1/Wi‑Fi 7/Bluetooth 6 products task.
    Notes:
    - Root node is set to non-critical to allow partial credit across products.
    - Networking 'thread' verification is implemented as a separate non-critical leaf due to framework constraints
      that disallow non-critical children under a critical parent.
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
        default_model=model
    )

    # 1) Extract products from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_products(),
        template_class=ProductsExtraction,
        extraction_name="products_extraction"
    )

    # 2) Keep only the first 4, pad if fewer
    products: List[ProductItem] = list(extracted.products[:4])
    while len(products) < 4:
        products.append(ProductItem())

    # 3) Build verification tree per product
    # Process products sequentially to ensure URL check happens before dependent verifications
    for i in range(4):
        await verify_product(evaluator, root, products[i], index_one_based=i + 1)

    # 4) Return standardized summary
    return evaluator.get_summary()