import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gaming_laptop_setup_solution"
TASK_DESCRIPTION = (
    "One complete gaming laptop setup purchase solution that satisfies all laptop, accessory, retailer, "
    "and required link/detail requirements."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProductItem(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None
    url: Optional[str] = None
    price: Optional[str] = None


class RetailerInfo(BaseModel):
    retailer_name: Optional[str] = None
    return_policy_url: Optional[str] = None
    warranty_url: Optional[str] = None
    financing_url: Optional[str] = None
    same_day_url: Optional[str] = None


class SolutionExtraction(BaseModel):
    retailer: Optional[RetailerInfo] = None
    laptop: Optional[ProductItem] = None
    cooling_pad: Optional[ProductItem] = None
    monitor: Optional[ProductItem] = None
    docking_station: Optional[ProductItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_solution() -> str:
    return """
    Extract exactly ONE complete purchase solution from the answer that includes:
    - retailer: 
        retailer_name
        return_policy_url (direct link to the retailer's return policy page)
        warranty_url (direct link to the retailer's extended warranty/protection plan options page)
        financing_url (direct link to the retailer's financing terms or credit card financing page)
        same_day_url (direct link to a page that confirms same-day pickup OR same-day delivery, if provided)
    - laptop (gaming laptop):
        name
        model
        url (direct link to the product page on the retailer's website)
        price (current price as a string)
    - cooling_pad:
        name
        model
        url (direct link to the product page on the retailer's website)
        price (current price as a string)
    - monitor (external gaming monitor):
        name
        model
        url (direct link to the product page on the retailer's website)
        price (current price as a string)
    - docking_station:
        name
        model
        url (direct link to the product page on the retailer's website)
        price (current price as a string)

    IMPORTANT:
    - If the answer presents multiple options, select the FIRST complete set and extract only that set.
    - Extract URLs exactly as they appear in the answer (if missing protocol, prepend http:// as needed).
    - If any field is missing in the answer, return null for that field.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(s: Optional[str]) -> str:
    return (s or "").strip()


def extract_domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        # Strip leading 'www.'
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return None


def is_url_from_retailer(url: Optional[str], retailer_domain: Optional[str]) -> bool:
    if not url or not retailer_domain:
        return False
    domain = extract_domain(url)
    return domain is not None and retailer_domain in domain


def infer_retailer_domain(extracted: SolutionExtraction) -> Optional[str]:
    # Prefer laptop URL to infer retailer domain; otherwise use retailer policy URLs
    candidates = [
        extracted.laptop.url if extracted.laptop else None,
        extracted.retailer.return_policy_url if extracted.retailer else None,
        extracted.retailer.warranty_url if extracted.retailer else None,
        extracted.retailer.financing_url if extracted.retailer else None,
    ]
    for u in candidates:
        d = extract_domain(u)
        if d:
            return d
    return None


def is_major_us_electronics_retailer(retailer_name: Optional[str], retailer_domain: Optional[str]) -> bool:
    """Heuristic check using known major US electronics retailers."""
    known_domains = {
        "bestbuy.com",
        "microcenter.com",
        "bhphotovideo.com",
        "newegg.com",
    }
    known_names = {"best buy", "micro center", "b&h", "b&h photo", "b&h photo video", "newegg"}

    name = (_safe_str(retailer_name)).lower()
    domain = (_safe_str(retailer_domain)).lower()

    domain_ok = any(d in domain for d in known_domains)
    name_ok = any(k in name for k in known_names)

    return domain_ok or name_ok


def build_all_product_urls(extracted: SolutionExtraction) -> List[str]:
    urls: List[str] = []
    if extracted.laptop and extracted.laptop.url:
        urls.append(extracted.laptop.url)
    if extracted.cooling_pad and extracted.cooling_pad.url:
        urls.append(extracted.cooling_pad.url)
    if extracted.monitor and extracted.monitor.url:
        urls.append(extracted.monitor.url)
    if extracted.docking_station and extracted.docking_station.url:
        urls.append(extracted.docking_station.url)
    return urls


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_retailer(
    evaluator: Evaluator,
    parent_node,
    extracted: SolutionExtraction,
) -> None:
    retailer = extracted.retailer or RetailerInfo()
    retailer_domain = infer_retailer_domain(extracted)
    product_urls = build_all_product_urls(extracted)

    # Retailer requirements node (critical)
    rr_node = evaluator.add_parallel(
        id="retailer_requirements",
        desc="Retailer identified and satisfies all retailer constraints, with required policy/terms links",
        parent=parent_node,
        critical=True,
    )

    # Retailer identification (critical existence)
    evaluator.add_custom_node(
        result=bool(_safe_str(retailer.retailer_name)),
        id="retailer_identification",
        desc="Retailer name provided",
        parent=rr_node,
        critical=True,
    )

    # Retailer is major U.S. electronics retailer (critical custom check)
    evaluator.add_custom_node(
        result=is_major_us_electronics_retailer(retailer.retailer_name, retailer_domain),
        id="retailer_major_us_electronics",
        desc="Retailer is a major U.S. electronics retailer (as required by the question)",
        parent=rr_node,
        critical=True,
    )

    # Return policy (critical, with two children)
    rp_node = evaluator.add_parallel(
        id="return_policy",
        desc="Return policy meets requirement and return policy link is provided",
        parent=rr_node,
        critical=True,
    )

    # Return policy URL existence & domain match
    evaluator.add_custom_node(
        result=bool(_safe_str(retailer.return_policy_url)) and is_url_from_retailer(retailer.return_policy_url, retailer_domain),
        id="return_policy_url",
        desc="Direct link to retailer return policy page provided",
        parent=rp_node,
        critical=True,
    )

    # Return period >= 15 days (verify by URL)
    rp_period_leaf = evaluator.add_leaf(
        id="return_period",
        desc="Standard return policy is at least 15 days",
        parent=rp_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The standard return policy period is at least 15 days.",
        node=rp_period_leaf,
        sources=retailer.return_policy_url,
        additional_instruction="Look for language like '15-day return policy' or any longer period (e.g., 30 days). If multiple categories differ, the standard/typical consumer electronics policy should be at least 15 days.",
    )

    # Extended warranty (critical, with two children)
    ew_node = evaluator.add_parallel(
        id="extended_warranty",
        desc="Extended warranty/protection plan with accidental damage coverage is available and link is provided",
        parent=rr_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_safe_str(retailer.warranty_url)) and is_url_from_retailer(retailer.warranty_url, retailer_domain),
        id="warranty_url",
        desc="Direct link to retailer extended warranty/protection plan options page provided",
        parent=ew_node,
        critical=True,
    )

    warranty_ad_leaf = evaluator.add_leaf(
        id="warranty_accidental_damage",
        desc="Warranty/protection plan includes accidental damage coverage",
        parent=ew_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The extended warranty/protection plan includes accidental damage coverage.",
        node=warranty_ad_leaf,
        sources=retailer.warranty_url,
        additional_instruction="Look for phrasing such as 'Accidental Damage', 'ADH', 'drops/spills' coverage. The plan should explicitly include accidental damage.",
    )

    # Financing options (critical, with two children)
    fin_node = evaluator.add_parallel(
        id="financing_options",
        desc="0% APR financing is available and financing terms link is provided",
        parent=rr_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_safe_str(retailer.financing_url)) and is_url_from_retailer(retailer.financing_url, retailer_domain),
        id="financing_url",
        desc="Direct link to retailer financing terms page provided",
        parent=fin_node,
        critical=True,
    )

    fin_apr_leaf = evaluator.add_leaf(
        id="financing_apr",
        desc="0% APR financing option available for qualified purchases",
        parent=fin_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This retailer offers 0% APR financing for qualified purchases.",
        node=fin_apr_leaf,
        sources=retailer.financing_url,
        additional_instruction="Accept phrasing like '0% APR', 'no interest if paid in full', or promotional 0% financing offers, typically above certain purchase thresholds (e.g., $599).",
    )

    # Same-day service (critical)
    sds_leaf = evaluator.add_leaf(
        id="same_day_service",
        desc="Same-day in-store pickup OR same-day delivery option is available (and confirmed)",
        parent=rr_node,
        critical=True,
    )
    sds_sources: List[str] = []
    if retailer.same_day_url:
        sds_sources.append(retailer.same_day_url)
    else:
        # Fall back to product URLs if no dedicated same-day link provided
        sds_sources = product_urls

    await evaluator.verify(
        claim="The retailer provides either same-day in-store pickup or same-day delivery options.",
        node=sds_leaf,
        sources=sds_sources if sds_sources else None,
        additional_instruction="Look for phrases like 'Same-Day Delivery', 'Pick up today', 'Same-day pickup'. Either one of same-day delivery or same-day in-store pickup must be available.",
    )


async def verify_laptop(
    evaluator: Evaluator,
    parent_node,
    extracted: SolutionExtraction,
) -> None:
    laptop = extracted.laptop or ProductItem()
    retailer_domain = infer_retailer_domain(extracted)

    # Gaming laptop node (critical)
    gl_node = evaluator.add_parallel(
        id="gaming_laptop",
        desc="Gaming laptop is provided with required details and meets all laptop technical requirements",
        parent=parent_node,
        critical=True,
    )

    # Product name and model provided (critical - from answer extraction)
    evaluator.add_custom_node(
        result=bool(_safe_str(laptop.name)) and bool(_safe_str(laptop.model)),
        id="laptop_product_name",
        desc="Laptop product name and model provided",
        parent=gl_node,
        critical=True,
    )

    # Direct URL provided and belongs to retailer (critical)
    evaluator.add_custom_node(
        result=bool(_safe_str(laptop.url)) and is_url_from_retailer(laptop.url, retailer_domain),
        id="laptop_url",
        desc="Direct laptop product page URL on the named retailer's website provided",
        parent=gl_node,
        critical=True,
    )

    # Price provided (critical)
    evaluator.add_custom_node(
        result=bool(_safe_str(laptop.price)),
        id="laptop_price",
        desc="Laptop current price provided",
        parent=gl_node,
        critical=True,
    )

    # Technical requirements (critical)
    ltr_node = evaluator.add_parallel(
        id="laptop_technical_requirements",
        desc="Laptop satisfies all specified technical requirements",
        parent=gl_node,
        critical=True,
    )

    # GPU: RTX 4060 or better
    gpu_leaf = evaluator.add_leaf(
        id="laptop_gpu",
        desc="GPU specification: NVIDIA GeForce RTX 4060 or better",
        parent=ltr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The laptop has an NVIDIA GeForce RTX 4060 or a higher-tier RTX 40-series GPU (e.g., RTX 4070/4080/4090).",
        node=gpu_leaf,
        sources=laptop.url,
        additional_instruction="Accept 'RTX 4060 Laptop GPU' and any higher model numbers (4070/4080/4090).",
    )

    # RAM: Minimum 16GB DDR5
    ram_leaf = evaluator.add_leaf(
        id="laptop_ram",
        desc="RAM specification: Minimum 16GB DDR5",
        parent=ltr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The laptop has at least 16GB of DDR5 memory.",
        node=ram_leaf,
        sources=laptop.url,
        additional_instruction="Look for '16GB DDR5' or a higher capacity. Accept configurations offering ≥16GB DDR5.",
    )

    # Display requirements (nested, critical)
    disp_node = evaluator.add_parallel(
        id="laptop_display",
        desc="Display meets size and refresh-rate requirements",
        parent=ltr_node,
        critical=True,
    )

    size_leaf = evaluator.add_leaf(
        id="display_size",
        desc="Display size is 15.6 inches",
        parent=disp_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The laptop's display size is 15.6 inches.",
        node=size_leaf,
        sources=laptop.url,
        additional_instruction="Allow minor formatting variants like '15.6-inch' or '15.6\"'.",
    )

    rr_leaf = evaluator.add_leaf(
        id="display_refresh_rate",
        desc="Refresh rate is 144Hz or 165Hz",
        parent=disp_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The laptop's display refresh rate is either 144Hz or 165Hz.",
        node=rr_leaf,
        sources=laptop.url,
        additional_instruction="Accept either 144 Hz or 165 Hz; minor formatting variants are acceptable.",
    )

    # Storage: Minimum 512GB SSD
    storage_leaf = evaluator.add_leaf(
        id="laptop_storage",
        desc="Storage specification: Minimum 512GB SSD",
        parent=ltr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The laptop has at least 512GB of SSD storage.",
        node=storage_leaf,
        sources=laptop.url,
        additional_instruction="Accept '512GB SSD' or any larger SSD capacity.",
    )

    # Battery: At least 70Wh
    battery_leaf = evaluator.add_leaf(
        id="laptop_battery",
        desc="Battery capacity: At least 70Wh",
        parent=ltr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The laptop's battery capacity is at least 70Wh.",
        node=battery_leaf,
        sources=laptop.url,
        additional_instruction="Look for values like 70Wh, 73Wh, 80Wh, etc. Any capacity ≥70Wh qualifies.",
    )

    # Ports: Thunderbolt 4 or USB-C with Power Delivery
    ports_leaf = evaluator.add_leaf(
        id="laptop_ports",
        desc="At least one Thunderbolt 4 or USB-C port with Power Delivery",
        parent=ltr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The laptop includes at least one Thunderbolt 4 port or a USB-C port with Power Delivery (PD).",
        node=ports_leaf,
        sources=laptop.url,
        additional_instruction="Accept 'Thunderbolt 4', 'USB-C with PD', or similar phrasing indicating PD capability.",
    )

    # RGB backlit keyboard
    kb_leaf = evaluator.add_leaf(
        id="laptop_keyboard",
        desc="RGB backlit keyboard",
        parent=ltr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The laptop features an RGB backlit keyboard.",
        node=kb_leaf,
        sources=laptop.url,
        additional_instruction="Accept variants like 'RGB-backlit keyboard', 'per-key RGB', or 'keyboard lighting RGB'.",
    )


async def verify_accessories(
    evaluator: Evaluator,
    parent_node,
    extracted: SolutionExtraction,
) -> None:
    retailer_domain = infer_retailer_domain(extracted)

    acc_node = evaluator.add_parallel(
        id="accessories",
        desc="All three required accessories are provided with required details and meet all accessory technical requirements",
        parent=parent_node,
        critical=True,
    )

    # Cooling pad group (critical)
    cp = extracted.cooling_pad or ProductItem()
    cp_node = evaluator.add_parallel(
        id="cooling_pad",
        desc="Cooling pad provided with required details and meets cooling pad requirements",
        parent=acc_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_safe_str(cp.name)) and bool(_safe_str(cp.model)),
        id="cooling_pad_product_name",
        desc="Cooling pad product name and model provided",
        parent=cp_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_safe_str(cp.url)) and is_url_from_retailer(cp.url, retailer_domain),
        id="cooling_pad_url",
        desc="Direct cooling pad product page URL on the named retailer's website provided",
        parent=cp_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_safe_str(cp.price)),
        id="cooling_pad_price",
        desc="Cooling pad current price provided",
        parent=cp_node,
        critical=True,
    )

    cp_req_node = evaluator.add_parallel(
        id="cooling_pad_technical_requirements",
        desc="Cooling pad satisfies all specified technical requirements",
        parent=cp_node,
        critical=True,
    )

    cp_size_leaf = evaluator.add_leaf(
        id="cooling_pad_size",
        desc="Cooling pad supports 15.6-inch laptops",
        parent=cp_req_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The cooling pad supports 15.6-inch laptops.",
        node=cp_size_leaf,
        sources=cp.url,
        additional_instruction="Look for 'supports up to 15.6\"' or similar phrasing. Accept ranges that include 15.6 inches.",
    )

    cp_fans_leaf = evaluator.add_leaf(
        id="cooling_pad_fans",
        desc="Cooling pad has at least 5 fans",
        parent=cp_req_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The cooling pad includes at least 5 fans.",
        node=cp_fans_leaf,
        sources=cp.url,
        additional_instruction="Look for specs indicating 5 or more fans.",
    )

    # External monitor group (critical)
    mon = extracted.monitor or ProductItem()
    mon_node = evaluator.add_parallel(
        id="external_monitor",
        desc="Monitor provided with required details and meets monitor requirements",
        parent=acc_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_safe_str(mon.name)) and bool(_safe_str(mon.model)),
        id="monitor_product_name",
        desc="Monitor product name and model provided",
        parent=mon_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_safe_str(mon.url)) and is_url_from_retailer(mon.url, retailer_domain),
        id="monitor_url",
        desc="Direct monitor product page URL on the named retailer's website provided",
        parent=mon_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_safe_str(mon.price)),
        id="monitor_price",
        desc="Monitor current price provided",
        parent=mon_node,
        critical=True,
    )

    mon_req_node = evaluator.add_parallel(
        id="monitor_technical_requirements",
        desc="Monitor satisfies all specified technical requirements",
        parent=mon_node,
        critical=True,
    )

    mon_size_leaf = evaluator.add_leaf(
        id="monitor_size",
        desc="Screen size is 27 inches",
        parent=mon_req_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The monitor has a 27-inch screen.",
        node=mon_size_leaf,
        sources=mon.url,
        additional_instruction="Accept '27-inch', '27\"'.",
    )

    mon_res_leaf = evaluator.add_leaf(
        id="monitor_resolution",
        desc="Resolution is 4K (3840x2160)",
        parent=mon_req_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The monitor resolution is 4K, specifically 3840×2160.",
        node=mon_res_leaf,
        sources=mon.url,
        additional_instruction="Accept '3840x2160', 'Ultra HD 4K'.",
    )

    mon_rr_leaf = evaluator.add_leaf(
        id="monitor_refresh_rate",
        desc="Refresh rate is at least 144Hz",
        parent=mon_req_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The monitor refresh rate is at least 144Hz.",
        node=mon_rr_leaf,
        sources=mon.url,
        additional_instruction="Accept 144Hz or any higher refresh rate.",
    )

    mon_rt_leaf = evaluator.add_leaf(
        id="monitor_response_time",
        desc="Response time is 1ms",
        parent=mon_req_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The monitor's response time is 1 ms.",
        node=mon_rt_leaf,
        sources=mon.url,
        additional_instruction="Accept '1ms', '1 ms (GtG/MPRT)' if clearly indicating 1ms.",
    )

    # Docking station group (critical)
    dock = extracted.docking_station or ProductItem()
    dock_node = evaluator.add_parallel(
        id="docking_station",
        desc="Dock provided with required details and meets dock requirements",
        parent=acc_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_safe_str(dock.name)) and bool(_safe_str(dock.model)),
        id="dock_product_name",
        desc="Docking station product name and model provided",
        parent=dock_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_safe_str(dock.url)) and is_url_from_retailer(dock.url, retailer_domain),
        id="dock_url",
        desc="Direct docking station product page URL on the named retailer's website provided",
        parent=dock_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_safe_str(dock.price)),
        id="dock_price",
        desc="Docking station current price provided",
        parent=dock_node,
        critical=True,
    )

    dock_req_node = evaluator.add_parallel(
        id="dock_technical_requirements",
        desc="Docking station satisfies all specified technical requirements",
        parent=dock_node,
        critical=True,
    )

    dock_conn_leaf = evaluator.add_leaf(
        id="dock_connectivity",
        desc="Supports USB-C or Thunderbolt 4 connection",
        parent=dock_req_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The docking station supports USB-C or Thunderbolt 4 connection.",
        node=dock_conn_leaf,
        sources=dock.url,
        additional_instruction="Look for 'USB-C' (with PD or Alt Mode) or 'Thunderbolt 4' compatibility.",
    )

    dock_dual_leaf = evaluator.add_leaf(
        id="dock_dual_monitor",
        desc="Supports dual monitor connectivity",
        parent=dock_req_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The docking station supports connections for two external monitors (dual-monitor).",
        node=dock_dual_leaf,
        sources=dock.url,
        additional_instruction="Look for 'dual display', 'connect two monitors', or specs indicating two video outputs.",
    )

    # Accessory compatibility (single critical leaf)
    compat_leaf = evaluator.add_leaf(
        id="accessory_compatibility",
        desc="All accessories are compatible with the selected laptop",
        parent=acc_node,
        critical=True,
    )
    compat_sources = []
    if extracted.laptop and extracted.laptop.url:
        compat_sources.append(extracted.laptop.url)
    if cp.url:
        compat_sources.append(cp.url)
    if dock.url:
        compat_sources.append(dock.url)
    if mon.url:
        compat_sources.append(mon.url)

    await evaluator.verify(
        claim=(
            "The selected accessories are compatible with the laptop: the cooling pad fits 15.6-inch laptops; "
            "the docking station connects via Thunderbolt 4 or USB-C (with PD/Alt Mode) and supports dual monitors; "
            "the external monitor uses standard video inputs that the dock/laptop can provide."
        ),
        node=compat_leaf,
        sources=compat_sources if compat_sources else None,
        additional_instruction=(
            "Check cross-compatibility: "
            "Cooling pad size includes 15.6\"; dock states compatibility with USB-C/TB4 laptops and dual display; "
            "monitor lists HDMI/DisplayPort or standard inputs that the dock provides. "
            "Minor brand-specific caveats are acceptable as long as general compatibility is indicated."
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
    Evaluate a single answer for the gaming laptop setup solution task and return a structured result dictionary.
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
        default_model=model,
    )

    # Extract solution details
    extracted = await evaluator.extract(
        prompt=prompt_extract_solution(),
        template_class=SolutionExtraction,
        extraction_name="complete_solution",
    )

    # Top-level critical solution node (to match rubric critical root)
    solution_node = evaluator.add_parallel(
        id="solution_root",
        desc="One complete gaming laptop setup purchase solution that satisfies all laptop, accessory, retailer, and required link/detail requirements",
        parent=root,
        critical=True,
    )

    # Build verification subtrees
    await verify_retailer(evaluator, solution_node, extracted)
    await verify_laptop(evaluator, solution_node, extracted)
    await verify_accessories(evaluator, solution_node, extracted)

    # Optional: record inferred retailer domain info
    inferred_domain = infer_retailer_domain(extracted)
    evaluator.add_custom_info(
        info={"inferred_retailer_domain": inferred_domain or "unknown"},
        info_type="domain_info",
        info_name="retailer_domain",
    )

    return evaluator.get_summary()