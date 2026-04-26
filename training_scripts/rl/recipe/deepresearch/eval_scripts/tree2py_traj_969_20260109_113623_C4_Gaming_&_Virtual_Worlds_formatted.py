import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "competitive_gaming_monitor"
TASK_DESCRIPTION = (
    "I'm building a competitive gaming setup for esports and need to find a suitable gaming monitor. "
    "Please identify one gaming monitor that meets the following professional-grade specifications:\n\n"
    "1. Refresh Rate: At least 144Hz\n"
    "2. Response Time: 1 millisecond (1ms) or less (GTG measurement)\n"
    "3. Screen Size: Between 24 and 27 inches (inclusive)\n"
    "4. Resolution: Minimum 1920×1080 (Full HD)\n"
    "5. Connectivity: Must include DisplayPort input\n"
    "6. Price: Under $500 USD\n"
    "7. Availability: Currently available for purchase in the United States\n\n"
    "For the monitor you identify, please provide:\n"
    "- The exact model name and manufacturer\n"
    "- All relevant specifications (refresh rate, response time, screen size, resolution, panel type, connectivity options)\n"
    "- The current retail price\n"
    "- A direct link to where it can be purchased or to its official product page"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MonitorExtraction(BaseModel):
    manufacturer: Optional[str] = None
    model_name: Optional[str] = None

    refresh_rate: Optional[str] = None  # e.g., "144Hz", "165 Hz"
    response_time: Optional[str] = None  # e.g., "1ms (GtG)", "0.5ms MPRT"
    response_time_method: Optional[str] = None  # e.g., "GTG", "G2G", "MPRT", "OD"
    screen_size: Optional[str] = None  # e.g., "24.5-inch", "27”"
    resolution: Optional[str] = None  # e.g., "1920x1080", "2560×1440 (QHD)"
    panel_type: Optional[str] = None  # e.g., "IPS", "TN", "VA"
    connectivity_options: List[str] = Field(default_factory=list)  # e.g., ["DisplayPort 1.2", "HDMI 2.0"]

    price: Optional[str] = None  # e.g., "$299.99", "USD 279"
    currency: Optional[str] = None  # e.g., "USD", "$"
    price_numeric_usd: Optional[str] = None  # if author provided a numeric parsing, otherwise null

    purchase_url: Optional[str] = None  # direct product/purchase page (preferred)
    official_url: Optional[str] = None  # official product page (alternative)
    additional_urls: List[str] = Field(default_factory=list)  # any other cited URLs

    # Any explicit statement about availability/location in the answer
    availability_statement: Optional[str] = None  # free text statement if present


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_monitor() -> str:
    return """
    Extract exactly one gaming monitor (pick the first one if multiple are presented or clearly choose the best-fitting one according to the constraints).
    Return all fields exactly as stated in the answer (do not invent or normalize beyond trivial cleanup). If a field is missing in the answer, set it to null or an empty list as appropriate.

    Required fields to extract:
    - manufacturer: The brand/manufacturer name (e.g., "ASUS", "Acer", "BenQ").
    - model_name: The exact model identifier (e.g., "VG259QM", "XL2546K").
    - refresh_rate: The stated refresh rate string (e.g., "144Hz", "240 Hz").
    - response_time: The stated response time string (e.g., "1ms (GtG)", "0.5ms MPRT").
    - response_time_method: If explicitly indicated, the method for response time (e.g., "GTG", "MPRT", "G2G"); otherwise null.
    - screen_size: The stated diagonal size string (e.g., "24.5-inch", "27”", "24.5\"").
    - resolution: The stated resolution string (e.g., "1920x1080", "2560×1440 (QHD)", "4K 3840x2160").
    - panel_type: Panel technology as stated (e.g., "IPS", "TN", "VA", "OLED").
    - connectivity_options: A list of connectivity interfaces mentioned (e.g., ["DisplayPort 1.2", "HDMI 2.0", "USB-C DP Alt Mode"]). If the answer lists connectivity in prose, extract individual items; if not listed, return [].

    - price: The monitor's stated current retail price as a string from the answer (e.g., "$299.99").
    - currency: Extract the currency code/symbol if given (e.g., "USD", "$"), else null.
    - price_numeric_usd: If the answer explicitly provides a numeric USD price (e.g., 279 or 279.99) or an easily parsed USD value, return it as a string; otherwise null.

    URLs (these must be explicitly present in the answer text; do not infer):
    - purchase_url: A direct link to a purchase page (Amazon, Best Buy, Micro Center, Newegg, B&H, etc.) if provided; else null.
    - official_url: The official product/brand page URL if provided; else null.
    - additional_urls: Any other URLs cited that refer to the product or relevant specs. Include all extras beyond purchase_url/official_url.

    - availability_statement: If the answer explicitly mentions that it's available in the US (e.g., “available in the United States” / “ships to US” / “US store”), copy the statement; else null.

    SPECIAL RULES FOR URL FIELDS:
    - Only extract URLs explicitly shown in the answer (plain URLs or markdown links).
    - Extract the full URL including protocol.
    - If a URL appears without protocol, prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def gather_all_urls(mon: MonitorExtraction) -> List[str]:
    urls: List[str] = []
    if _nonempty(mon.purchase_url):
        urls.append(mon.purchase_url.strip())
    if _nonempty(mon.official_url):
        urls.append(mon.official_url.strip())
    for u in mon.additional_urls:
        if _nonempty(u):
            urls.append(u.strip())
    # de-duplicate while preserving order
    seen = set()
    result = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def monitor_full_name(mon: MonitorExtraction) -> str:
    mfg = mon.manufacturer.strip() if _nonempty(mon.manufacturer) else ""
    mdl = mon.model_name.strip() if _nonempty(mon.model_name) else ""
    return f"{mfg} {mdl}".strip()


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root, mon: MonitorExtraction) -> None:
    # For convenience
    sources_all = gather_all_urls(mon)

    # 1) Purchase or official product page URL (critical)
    url_node = evaluator.add_parallel(
        id="Purchase_Or_Official_Product_Page_URL",
        desc="Provides a direct URL to a purchase page or the official product page for the identified monitor.",
        parent=root,
        critical=True,
    )

    url_provided_leaf = evaluator.add_custom_node(
        result=_nonempty(mon.purchase_url) or _nonempty(mon.official_url),
        id="product_url_provided",
        desc="At least one direct product or official URL is provided in the answer.",
        parent=url_node,
        critical=True,
    )

    # Validate that at least one provided URL is a product page for this monitor (contains model/brand)
    url_valid_leaf = evaluator.add_leaf(
        id="product_url_valid_for_model",
        desc="Provided URL(s) correspond to the identified monitor model/manufacturer.",
        parent=url_node,
        critical=True,
    )
    claim_url_valid = (
        f"The provided product page(s) include or clearly correspond to the monitor '{monitor_full_name(mon)}'. "
        f"Minor variants in punctuation/case are acceptable."
    )
    await evaluator.verify(
        claim=claim_url_valid,
        node=url_valid_leaf,
        sources=sources_all,
        additional_instruction=(
            "Check if the page title or product details show the same model or an equivalent model number from the same brand. "
            "Allow minor spacing/casing or hyphen differences in model names. If no page clearly shows the model/brand, mark as unsupported."
        ),
    )

    # 2) Model and manufacturer (critical, with 'provided' + 'matches page' checks)
    model_node = evaluator.add_parallel(
        id="Model_And_Manufacturer_Provided",
        desc="Provides the exact monitor model name and the manufacturer/brand.",
        parent=root,
        critical=True,
    )

    model_provided_leaf = evaluator.add_custom_node(
        result=_nonempty(mon.model_name) and _nonempty(mon.manufacturer),
        id="model_manufacturer_provided",
        desc="The answer provides both the exact model name and the manufacturer.",
        parent=model_node,
        critical=True,
    )

    model_match_leaf = evaluator.add_leaf(
        id="model_brand_match_page",
        desc="The product page(s) reflect the same model and manufacturer.",
        parent=model_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The monitor on the provided page(s) is '{monitor_full_name(mon)}' (allow minor naming variants).",
        node=model_match_leaf,
        sources=sources_all,
        additional_instruction=(
            "Confirm that the page references the same brand and model as stated in the answer. "
            "Accept minor punctuation/spacing/case differences. If the page shows a different model, fail."
        ),
    )

    # 3) Refresh rate (critical) - Provided + Compliant
    rr_node = evaluator.add_parallel(
        id="Refresh_Rate_Provided_And_Compliant",
        desc="States the monitor refresh rate and it is at least 144Hz.",
        parent=root,
        critical=True,
    )
    rr_provided_leaf = evaluator.add_custom_node(
        result=_nonempty(mon.refresh_rate),
        id="refresh_rate_provided",
        desc="The answer states the monitor's refresh rate.",
        parent=rr_node,
        critical=True,
    )
    rr_compliant_leaf = evaluator.add_leaf(
        id="refresh_rate_compliant",
        desc="Monitor supports at least 144Hz refresh rate.",
        parent=rr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This monitor supports a refresh rate of at least 144Hz.",
        node=rr_compliant_leaf,
        sources=sources_all,
        additional_instruction=(
            "Check the specifications on the product/retailer page for refresh rate. "
            "Accept if max refresh rate is >= 144Hz (e.g., 144Hz, 165Hz, 240Hz)."
        ),
    )

    # 4) Response time (critical) - Provided + Compliant
    rt_node = evaluator.add_parallel(
        id="Response_Time_Provided_And_Compliant",
        desc="States the monitor response time and it is 1ms or less, using an allowed methodology as stated in the constraints (GTG or MPRT).",
        parent=root,
        critical=True,
    )
    rt_provided_leaf = evaluator.add_custom_node(
        result=_nonempty(mon.response_time),
        id="response_time_provided",
        desc="The answer states the monitor's response time.",
        parent=rt_node,
        critical=True,
    )
    rt_compliant_leaf = evaluator.add_leaf(
        id="response_time_compliant",
        desc="Monitor response time is 1ms or less (GTG or MPRT).",
        parent=rt_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This monitor's response time is 1 ms or less using either GtG (grey-to-grey) or MPRT measurement.",
        node=rt_compliant_leaf,
        sources=sources_all,
        additional_instruction=(
            "Check stated response time on the product/retailer page. "
            "Accept 1ms (GtG) or 1ms (MPRT) or anything clearly <= 1ms with an allowed methodology. "
            "If the page only shows >1ms (e.g., 2ms), fail."
        ),
    )

    # 5) Screen size (critical) - Provided + Compliant
    size_node = evaluator.add_parallel(
        id="Screen_Size_Provided_And_Compliant",
        desc="States the screen size and it is between 24 and 27 inches inclusive.",
        parent=root,
        critical=True,
    )
    size_provided_leaf = evaluator.add_custom_node(
        result=_nonempty(mon.screen_size),
        id="screen_size_provided",
        desc="The answer states the monitor's screen size.",
        parent=size_node,
        critical=True,
    )
    size_compliant_leaf = evaluator.add_leaf(
        id="screen_size_compliant",
        desc="Monitor screen size is between 24 and 27 inches inclusive.",
        parent=size_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This monitor has a screen size between 24.0 and 27.0 inches inclusive.",
        node=size_compliant_leaf,
        sources=sources_all,
        additional_instruction=(
            "Check the diagonal size in inches on the product page. "
            "Accept 24.0, 24.5, 25, 25.5, 26, 27 inches, etc., as long as it falls within [24, 27]. "
            "If listed in cm, convert approximately and judge accordingly."
        ),
    )

    # 6) Resolution (critical) - Provided + Compliant
    res_node = evaluator.add_parallel(
        id="Resolution_Provided_And_Compliant",
        desc="States the resolution and it is at least 1920×1080 (Full HD).",
        parent=root,
        critical=True,
    )
    res_provided_leaf = evaluator.add_custom_node(
        result=_nonempty(mon.resolution),
        id="resolution_provided",
        desc="The answer states the monitor's resolution.",
        parent=res_node,
        critical=True,
    )
    res_compliant_leaf = evaluator.add_leaf(
        id="resolution_compliant",
        desc="Monitor resolution is at least 1920×1080.",
        parent=res_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This monitor's native resolution is at least 1920×1080.",
        node=res_compliant_leaf,
        sources=sources_all,
        additional_instruction=(
            "Accept Full HD (1920×1080) or higher such as 2560×1440 (QHD) or 3840×2160 (4K). "
            "If resolution is below 1920×1080, fail."
        ),
    )

    # 7) Connectivity incl. DisplayPort (critical) - Provided + DP Compliant
    conn_node = evaluator.add_parallel(
        id="Connectivity_Options_Provided_Including_DisplayPort",
        desc="Provides connectivity options and confirms the monitor includes at least one DisplayPort input.",
        parent=root,
        critical=True,
    )
    conn_provided_leaf = evaluator.add_custom_node(
        result=(mon.connectivity_options is not None and len(mon.connectivity_options) > 0)
               or (mon.connectivity_options is not None and any(_nonempty(x) for x in mon.connectivity_options)),
        id="connectivity_provided",
        desc="The answer lists connectivity options.",
        parent=conn_node,
        critical=True,
    )
    dp_compliant_leaf = evaluator.add_leaf(
        id="displayport_included",
        desc="Monitor includes at least one DisplayPort input.",
        parent=conn_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This monitor includes at least one DisplayPort input (e.g., DP, DisplayPort 1.2/1.4/2.0).",
        node=dp_compliant_leaf,
        sources=sources_all,
        additional_instruction=(
            "Look for 'DisplayPort' or 'DP' in the I/O section. "
            "If only HDMI/USB-C is listed and no DisplayPort, fail."
        ),
    )

    # 8) Panel type provided (critical)
    panel_node = evaluator.add_parallel(
        id="Panel_Type_Provided",
        desc="States the panel type (e.g., IPS/TN/VA/etc.).",
        parent=root,
        critical=True,
    )
    panel_provided_leaf = evaluator.add_custom_node(
        result=_nonempty(mon.panel_type),
        id="panel_type_provided",
        desc="The answer states the panel technology.",
        parent=panel_node,
        critical=True,
    )

    # 9) Price provided and compliant (critical) - Provided + Under $500
    price_node = evaluator.add_parallel(
        id="Price_Provided_And_Compliant",
        desc="Provides the current retail price and it is under $500 USD.",
        parent=root,
        critical=True,
    )
    price_provided_leaf = evaluator.add_custom_node(
        result=_nonempty(mon.price),
        id="price_provided",
        desc="The answer states a current retail price.",
        parent=price_node,
        critical=True,
    )
    price_compliant_leaf = evaluator.add_leaf(
        id="price_under_500usd",
        desc="Monitor price is under $500 USD at the referenced page(s).",
        parent=price_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The current purchase price for this monitor is under $500 USD.",
        node=price_compliant_leaf,
        sources=sources_all,
        additional_instruction=(
            "Check the product or retailer page for the live price. "
            "Use USD price if shown; if multiple prices, prioritize the base product price (exclude shipping/warranties). "
            "If price equals or exceeds $500, fail. Accept minor rounding differences (e.g., $499.99 is under $500)."
        ),
    )

    # 10) US availability (critical)
    us_node = evaluator.add_parallel(
        id="US_Availability_Compliant",
        desc="Indicates the monitor is currently available for purchase in the United States.",
        parent=root,
        critical=True,
    )
    us_avail_leaf = evaluator.add_leaf(
        id="us_availability_now",
        desc="The product page indicates current availability to purchase in the US.",
        parent=us_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This monitor is currently available for purchase in the United States.",
        node=us_avail_leaf,
        sources=sources_all,
        additional_instruction=(
            "Evidence may include: USD pricing on a US retailer site (e.g., Amazon.com, BestBuy.com, Newegg.com), "
            "US-specific product page, 'in stock' indicators, or shipping to US. "
            "If page shows 'out of stock' or region not US, fail."
        ),
    )

    # 11) Preferred panel technology IPS or TN (non-critical)
    preferred_node = evaluator.add_parallel(
        id="Preferred_Panel_Technology_IPS_or_TN",
        desc="Panel technology is IPS or TN (preferred but not required).",
        parent=root,
        critical=False,
    )
    preferred_leaf = evaluator.add_leaf(
        id="panel_is_ips_or_tn",
        desc="The monitor uses IPS or TN panel technology.",
        parent=preferred_node,
        critical=False,
    )
    await evaluator.verify(
        claim="The monitor's panel technology is either IPS or TN.",
        node=preferred_leaf,
        sources=sources_all,
        additional_instruction=(
            "If the page shows IPS or TN explicitly, pass. If VA/OLED/Mini-LED/etc., fail this non-critical check."
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel; critical children gate overall
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

    # Extract a single monitor from the answer
    extracted_monitor = await evaluator.extract(
        prompt=prompt_extract_monitor(),
        template_class=MonitorExtraction,
        extraction_name="monitor_extraction",
    )

    # Build verification nodes and run checks
    await build_verification_tree(evaluator, root, extracted_monitor)

    return evaluator.get_summary()