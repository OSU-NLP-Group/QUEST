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
TASK_ID = "wifi7_router_specs"
TASK_DESCRIPTION = """I am researching WiFi 7 routers for a high-performance home network upgrade and need to compare options that meet specific technical requirements. Find three distinct WiFi 7 router models currently available for purchase that each meet all of the following specifications:

1. Support 320 MHz channel bandwidth on the 6 GHz band
2. Include at least one 10 Gigabit Ethernet port
3. Have a specified coverage area in square feet
4. Have a specified maximum number of simultaneous device connections supported
5. Operate on all three frequency bands: 2.4 GHz, 5 GHz, and 6 GHz (tri-band)

For each of the three router models, provide:
- The complete model name and manufacturer
- Confirmation that it supports 320 MHz channel bandwidth on 6 GHz
- The number of 10 Gigabit Ethernet ports included
- The coverage area in square feet
- The maximum number of simultaneous devices supported
- A direct URL to the product page (manufacturer or major retailer website)
- The current retail price in USD

All three routers must be different models and meet all the specified requirements.
"""

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class RouterSpec(BaseModel):
    """Structured information for one router as extracted from the answer."""
    model_name: Optional[str] = None
    manufacturer: Optional[str] = None
    product_url: Optional[str] = None
    price_usd: Optional[str] = None

    # Requirement-specific fields extracted from the answer
    support_320mhz_6ghz: Optional[str] = None  # e.g., "yes", "supports 320 MHz", "no", or null
    ten_g_ports_count: Optional[str] = None    # e.g., "1", "2", "1x RJ45 + 1x SFP+", etc.
    coverage_area_sqft: Optional[str] = None   # e.g., "3000", "3000 sq ft", "up to 3000 sq ft"
    max_devices_supported: Optional[str] = None  # e.g., "200", "up to 200 devices"
    tri_band_bands: Optional[str] = None       # e.g., "2.4 GHz, 5 GHz, 6 GHz" or "tri-band 2.4/5/6 GHz"

    # Additional URLs that the answer may cite (beyond the main product_url)
    supporting_urls: List[str] = Field(default_factory=list)


class RoutersExtraction(BaseModel):
    """Top-level extracted list of routers."""
    routers: List[RouterSpec] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_routers() -> str:
    return """
    Extract up to three WiFi 7 router entries described in the answer. Each router must be represented with the following fields as explicitly stated in the answer text. Do not invent information.

    For each router, extract:
    - model_name: The complete router model name (e.g., "ASUS RT-BE88U")
    - manufacturer: The brand/manufacturer (e.g., "ASUS")
    - product_url: A direct URL to the product page on the manufacturer's website or a major retailer site (Amazon, Best Buy, Micro Center, Newegg, Walmart, B&H, etc.). Extract only URLs explicitly present in the answer; include the protocol (http/https).
    - price_usd: The current retail price in USD as stated in the answer (keep the string exactly as shown, e.g., "$599.99" or "USD 599")
    - support_320mhz_6ghz: Whether the answer explicitly states support for 320 MHz on the 6 GHz band (e.g., "supports 320 MHz on 6 GHz"). If stated, extract a short confirming phrase (e.g., "supports 320 MHz on 6 GHz"). If not stated, return null.
    - ten_g_ports_count: The number (or textual description) of 10 Gigabit Ethernet ports. If the answer only says "at least one 10G port", extract that phrase. If not stated, return null.
    - coverage_area_sqft: The coverage area in square feet as stated (e.g., "3000 sq ft", "up to 3000 sq ft"). If not stated, return null.
    - max_devices_supported: The maximum number of simultaneous devices supported, as stated (e.g., "up to 200 devices"). If not stated, return null.
    - tri_band_bands: A confirmation string that the router operates on 2.4 GHz, 5 GHz, and 6 GHz (tri-band), e.g., "2.4 GHz / 5 GHz / 6 GHz tri-band". If not stated, return null.
    - supporting_urls: Any additional URLs cited in the answer for this router's specs (beyond product_url). Extract only URLs explicitly present.

    Rules:
    - Do not infer; only extract what is explicitly in the answer text.
    - If the answer includes more than three routers, extract the first three mentioned.
    - If any field is missing for a router, set it to null (except supporting_urls which should be an empty array if none).
    - For URLs, accept plain URLs or markdown links; always extract the actual link.
    - Do not modify numbers or units; return them exactly as presented.

    Return a JSON object with a single key 'routers' that is an array of at most 3 RouterSpec objects.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_model_key(router: RouterSpec) -> Optional[str]:
    """Create a normalized key to compare distinctness of manufacturer+model."""
    if not router.model_name or not router.manufacturer:
        return None
    key = f"{router.manufacturer.strip().lower()}::{router.model_name.strip().lower()}"
    return key


def _get_all_sources(router: RouterSpec) -> List[str]:
    """Combine product URL with any supporting URLs."""
    sources: List[str] = []
    if router.product_url and router.product_url.strip():
        sources.append(router.product_url.strip())
    for u in router.supporting_urls:
        if u and isinstance(u, str) and u.strip():
            sources.append(u.strip())
    return sources


def _has_text(value: Optional[str]) -> bool:
    return bool(value and isinstance(value, str) and value.strip())


# --------------------------------------------------------------------------- #
# Verification logic per router                                               #
# --------------------------------------------------------------------------- #
async def verify_router(
    evaluator: Evaluator,
    parent_node,
    router: RouterSpec,
    router_index: int,
) -> None:
    """
    Build and verify all required leaf checks for a single router.
    Each leaf is critical under this router node.
    """
    idx_display = router_index + 1
    router_node = evaluator.add_parallel(
        id=f"Router_{idx_display}",
        desc=f"WiFi 7 Router #{idx_display} verification with complete specifications",
        parent=parent_node,
        critical=True  # Critical: failing any router fails the overall task requirement
    )

    # 1) Product URL validity and model-manufacturer match
    url_leaf = evaluator.add_leaf(
        id=f"Router_{idx_display}_Product_URL",
        desc="A valid manufacturer or retailer URL for the specific router model is provided",
        parent=router_node,
        critical=True
    )
    if _has_text(router.product_url) and _has_text(router.model_name) and _has_text(router.manufacturer):
        claim = f"This URL is a product page for the router model '{router.model_name}' by '{router.manufacturer}'."
        sources = router.product_url
        add_ins = (
            "Verify that the page is a product detail page (manufacturer site or major retailer). "
            "Confirm the page title or details include the model name and brand/manufacturer."
        )
        await evaluator.verify(claim=claim, node=url_leaf, sources=sources, additional_instruction=add_ins)
    else:
        # Missing essential info implies failure of this leaf
        url_leaf.score = 0.0
        url_leaf.status = "failed"

    # Prepare combined sources for subsequent checks
    all_sources = _get_all_sources(router)

    # 2) 320 MHz on 6 GHz support
    mhz_leaf = evaluator.add_leaf(
        id=f"Router_{idx_display}_320MHz_Support",
        desc="Router supports 320 MHz channel bandwidth on 6 GHz band",
        parent=router_node,
        critical=True
    )
    if _has_text(router.support_320mhz_6ghz) and all_sources:
        claim = "This router supports 320 MHz channel bandwidth on the 6 GHz band."
        add_ins = (
            "Look for text such as '320 MHz', '320MHz', 'channel width 320 MHz', "
            "or 'supports 320 MHz on 6 GHz'. Allow reasonable phrasing variants."
        )
        await evaluator.verify(claim=claim, node=mhz_leaf, sources=all_sources, additional_instruction=add_ins)
    else:
        mhz_leaf.score = 0.0
        mhz_leaf.status = "failed"

    # 3) At least one 10 Gigabit Ethernet port
    ten_g_leaf = evaluator.add_leaf(
        id=f"Router_{idx_display}_10G_Ethernet",
        desc="Router includes at least one 10 Gigabit Ethernet port",
        parent=router_node,
        critical=True
    )
    if _has_text(router.ten_g_ports_count) and all_sources:
        claim = (
            "This router includes at least one 10 Gigabit Ethernet port "
            "(RJ45 10GBASE-T or 10G SFP+ qualifies)."
        )
        add_ins = (
            "Accept descriptions like '10GbE', '10G WAN/LAN', '10G RJ45', '10G SFP+', or similar. "
            "At least one 10G-capable Ethernet port must be present."
        )
        await evaluator.verify(claim=claim, node=ten_g_leaf, sources=all_sources, additional_instruction=add_ins)
    else:
        ten_g_leaf.score = 0.0
        ten_g_leaf.status = "failed"

    # 4) Coverage area in square feet is specified
    cov_leaf = evaluator.add_leaf(
        id=f"Router_{idx_display}_Coverage_Area",
        desc="Router's coverage area in square feet is specified",
        parent=router_node,
        critical=True
    )
    if _has_text(router.coverage_area_sqft) and all_sources:
        # Use the exact string from the answer to ground the claim
        cov_val = router.coverage_area_sqft.strip()
        claim = f"The router's coverage area is {cov_val} (in square feet)."
        add_ins = (
            "Verify coverage phrasing such as 'coverage up to X sq ft', 'X square feet', etc. "
            "The page must explicitly mention a coverage area in square feet."
        )
        await evaluator.verify(claim=claim, node=cov_leaf, sources=all_sources, additional_instruction=add_ins)
    else:
        cov_leaf.score = 0.0
        cov_leaf.status = "failed"

    # 5) Maximum number of simultaneous devices supported is specified
    dev_leaf = evaluator.add_leaf(
        id=f"Router_{idx_display}_Device_Capacity",
        desc="Maximum number of supported simultaneous device connections is specified",
        parent=router_node,
        critical=True
    )
    if _has_text(router.max_devices_supported) and all_sources:
        dev_val = router.max_devices_supported.strip()
        claim = f"The router supports up to {dev_val} devices simultaneously."
        add_ins = (
            "Look for phrases like 'supports up to X devices', 'connected devices', or 'max devices'. "
            "The page must state an explicit maximum number of devices."
        )
        await evaluator.verify(claim=claim, node=dev_leaf, sources=all_sources, additional_instruction=add_ins)
    else:
        dev_leaf.score = 0.0
        dev_leaf.status = "failed"

    # 6) Tri-band operation on 2.4/5/6 GHz
    tri_leaf = evaluator.add_leaf(
        id=f"Router_{idx_display}_Tri_Band",
        desc="Router operates on all three bands: 2.4 GHz, 5 GHz, and 6 GHz",
        parent=router_node,
        critical=True
    )
    if _has_text(router.tri_band_bands) and all_sources:
        claim = "This router is tri-band, operating on 2.4 GHz, 5 GHz, and 6 GHz."
        add_ins = (
            "Verify the page mentions all three bands (2.4 GHz, 5 GHz, and 6 GHz). "
            "Allow reasonable phrasing variants, e.g., 'tri-band 2.4/5/6 GHz'."
        )
        await evaluator.verify(claim=claim, node=tri_leaf, sources=all_sources, additional_instruction=add_ins)
    else:
        tri_leaf.score = 0.0
        tri_leaf.status = "failed"

    # 7) Current retail price in USD is provided
    price_leaf = evaluator.add_leaf(
        id=f"Router_{idx_display}_Price",
        desc="Current retail price of the router is provided",
        parent=router_node,
        critical=True
    )
    if _has_text(router.price_usd) and all_sources:
        price_val = router.price_usd.strip()
        claim = f"The current retail price is {price_val} USD."
        add_ins = (
            "Confirm the price shown on the product page or major retailer page is in USD and matches (or is very close to) the stated price. "
            "Minor variations may be acceptable due to discounts or regional differences."
        )
        await evaluator.verify(claim=claim, node=price_leaf, sources=all_sources, additional_instruction=add_ins)
    else:
        price_leaf.score = 0.0
        price_leaf.status = "failed"


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the answer for the WiFi 7 router comparison task.
    Builds a verification tree according to the rubric and returns a summary dict.
    """
    # Initialize evaluator; root node as non-critical parallel aggregator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Evaluate whether three distinct WiFi 7 router models meeting specified technical requirements have been identified with complete specifications",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract router entries from the answer
    routers_extracted = await evaluator.extract(
        prompt=prompt_extract_routers(),
        template_class=RoutersExtraction,
        extraction_name="routers_extraction",
    )

    # Normalize to exactly three entries (pad with empty placeholders if needed)
    routers_list: List[RouterSpec] = list(routers_extracted.routers[:3])
    while len(routers_list) < 3:
        routers_list.append(RouterSpec())

    # Add distinctness check (critical)
    names_keys = [_normalize_model_key(r) for r in routers_list]
    # Distinctness: all keys must be present and unique
    distinct = all(k is not None for k in names_keys) and len(set(names_keys)) == 3

    evaluator.add_custom_node(
        result=distinct,
        id="Routers_Distinctness",
        desc="All three router models are distinct (no duplicates among Router 1, Router 2, and Router 3)",
        parent=root,
        critical=True
    )

    # Build Router 1/2/3 subtrees (each critical)
    for i in range(3):
        await verify_router(evaluator, root, routers_list[i], i)

    # Final summary
    return evaluator.get_summary()