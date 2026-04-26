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
TASK_ID = "eu_usb_c_compliance_analysis_2025_11_30"
TASK_DESCRIPTION = (
    "As of November 30, 2025, the European Union has implemented regulations requiring USB Type-C charging ports "
    "on most portable electronic devices. Create a compliance analysis report for four consumer electronic devices "
    "currently available in the European market, selecting one device from each of the following categories: "
    "(1) smartphone, (2) laptop, (3) wireless earbuds, and (4) handheld gaming console.\n\n"
    "For each device, your analysis must include:\n\n"
    "1. Device Category Classification: Confirm the device belongs to the specified category and verify whether this "
    "category is covered under the EU USB-C mandate, including the applicable effective date.\n"
    "2. Charging Port Specification: Identify the device's charging port type (e.g., USB-C, Lightning, Micro-USB, proprietary connector) "
    "with supporting technical specifications.\n"
    "3. Power Consumption Analysis: Specify the device's charging power consumption in watts and assess whether it falls "
    "under the 100-watt threshold that determines mandate applicability.\n"
    "4. Exemption Eligibility: Evaluate whether the device qualifies for any exemptions from the mandate, specifically checking:\n"
    "   - Whether the device is too small to physically accommodate a USB-C port\n"
    "   - Whether the device charges exclusively through wireless means (no wired charging capability)\n"
    "5. Compliance Status Determination: Based on the above factors, determine:\n"
    "   - The applicable compliance deadline for this device category\n"
    "   - Whether the device currently meets the EU USB-C charging requirement\n"
    "   - The device's overall compliance status\n\n"
    "Each piece of information must be supported by reference URLs from reliable sources (official manufacturer specifications, "
    "EU regulatory sources, or reputable technology publications). Ensure your analysis considers the correct effective dates: "
    "December 28, 2024, for most device categories, and April 28, 2026, for laptops."
)

AS_OF_DATE = "Nov 30, 2025"
MANDATE_DEADLINES = {
    "smartphone": "December 28, 2024",
    "wireless earbuds": "December 28, 2024",
    "handheld gaming console": "December 28, 2024",
    "laptop": "April 28, 2026",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DeviceComplianceItem(BaseModel):
    """Structured compliance information for a single device."""
    model_name: Optional[str] = None
    category: Optional[str] = None

    # General product info sources (e.g., manufacturer product page)
    model_info_urls: List[str] = Field(default_factory=list)

    # EU market availability sources (EU retailers, EU region product pages, etc.)
    eu_availability_urls: List[str] = Field(default_factory=list)

    # EU regulatory source URLs confirming category coverage and effective date
    eu_category_urls: List[str] = Field(default_factory=list)

    # Charging port information
    charging_port_type: Optional[str] = None
    charging_port_urls: List[str] = Field(default_factory=list)

    # Charging power information
    charging_power_watts: Optional[str] = None
    charging_power_urls: List[str] = Field(default_factory=list)

    # Exemption checks
    exemption_too_small: Optional[str] = None  # "yes"/"no"/"unknown"
    exemption_wireless_only: Optional[str] = None  # "yes"/"no"/"unknown"
    exemption_urls: List[str] = Field(default_factory=list)

    # USB Power Delivery support
    usb_pd_support: Optional[str] = None  # "yes"/"no"/"unknown"
    usb_pd_urls: List[str] = Field(default_factory=list)

    # Compliance determination
    compliance_deadline: Optional[str] = None
    compliance_meets_requirement: Optional[str] = None  # "yes"/"no"/"unknown"
    compliance_status: Optional[str] = None
    compliance_urls: List[str] = Field(default_factory=list)


class DevicesExtraction(BaseModel):
    """Extraction for all required device categories."""
    smartphone: Optional[DeviceComplianceItem] = None
    laptop: Optional[DeviceComplianceItem] = None
    wireless_earbuds: Optional[DeviceComplianceItem] = None
    handheld_gaming_console: Optional[DeviceComplianceItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_devices() -> str:
    return (
        "Extract compliance analysis data for exactly four devices currently available in the EU market, one per category:\n"
        "1) smartphone\n"
        "2) laptop\n"
        "3) wireless earbuds\n"
        "4) handheld gaming console\n\n"
        "For each category, extract a single representative device's information with the following fields:\n"
        "- model_name: The specific model identified in the answer.\n"
        "- category: The category label used in the answer (e.g., 'smartphone', 'laptop', 'wireless earbuds', 'handheld gaming console').\n"
        "- model_info_urls: URLs to manufacturer product pages or reputable specifications articles.\n"
        "- eu_availability_urls: URLs indicating the device is sold/available in the EU market (manufacturer EU pages or EU retailers).\n"
        "- eu_category_urls: URLs to EU regulatory sources confirming this category is covered and its effective date.\n"
        "- charging_port_type: The charging connector type (e.g., 'USB-C', 'Lightning', 'Micro-USB', 'proprietary').\n"
        "- charging_port_urls: URLs supporting the stated charging port type.\n"
        "- charging_power_watts: The charging power (in watts) as stated in the answer (allow ranges or approximate values as text).\n"
        "- charging_power_urls: URLs supporting the stated charging power figure.\n"
        "- exemption_too_small: 'yes' if the device is claimed too small to physically accommodate USB-C; 'no' otherwise; 'unknown' if not determined.\n"
        "- exemption_wireless_only: 'yes' if the device is claimed to charge exclusively through wireless means; 'no' otherwise; 'unknown' if not determined.\n"
        "- exemption_urls: URLs supporting the exemption assessments (dimensions, charging capabilities).\n"
        "- usb_pd_support: 'yes' if USB PD support is claimed; 'no' otherwise; 'unknown' if not determined.\n"
        "- usb_pd_urls: URLs supporting USB PD support claims.\n"
        "- compliance_deadline: The deadline date mentioned in the answer for this category (string).\n"
        "- compliance_meets_requirement: 'yes'/'no'/'unknown' according to the answer's determination.\n"
        "- compliance_status: A brief status string (e.g., 'Compliant', 'Non-compliant', 'Not required yet').\n"
        "- compliance_urls: URLs used to support the compliance determination.\n\n"
        "Return a JSON object with fields: smartphone, laptop, wireless_earbuds, handheld_gaming_console.\n"
        "If any field is missing for a device, set it to null for strings or an empty array for URLs.\n"
        "Extract only URLs explicitly mentioned in the answer; accept plain URLs or markdown links. Include full URLs."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) else ""

def _dedup_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    result = []
    for lst in url_lists:
        for u in lst:
            if not isinstance(u, str):
                continue
            u2 = u.strip()
            if u2 and u2 not in seen:
                seen.add(u2)
                result.append(u2)
    return result

def _parse_yes_no(val: Optional[str]) -> Optional[bool]:
    if val is None:
        return None
    s = val.strip().lower()
    if s in {"yes", "true", "y", "supported", "support", "supports"}:
        return True
    if s in {"no", "false", "n", "not supported", "does not support", "unsupported"}:
        return False
    return None

def _extract_max_watts(text: Optional[str]) -> Optional[float]:
    """Extract the maximum numeric watt value from a free-form string, if any."""
    if not text or not isinstance(text, str):
        return None
    import re
    nums = re.findall(r"(\d+(?:\.\d+)?)", text)
    if not nums:
        return None
    try:
        vals = [float(x) for x in nums]
        return max(vals) if vals else None
    except Exception:
        return None

def _expected_deadline_for_category(expected_category: str) -> str:
    key = expected_category.strip().lower()
    # Use mapping; fallback to December 28, 2024 for covered portable device categories
    for k, v in MANDATE_DEADLINES.items():
        if key == k:
            return v
    return "December 28, 2024"


# --------------------------------------------------------------------------- #
# Verification for one device                                                 #
# --------------------------------------------------------------------------- #
async def verify_device(
    evaluator: Evaluator,
    parent_node,
    item: DeviceComplianceItem,
    prefix_id: str,
    device_desc: str,
    expected_category: str,
) -> None:
    """
    Build verification tree and run checks for a single device category.
    """
    # Device-level parent node
    device_node = evaluator.add_parallel(
        id=f"{prefix_id}_device",
        desc=device_desc,
        parent=parent_node,
        critical=False,
    )

    # -------- EU Market Availability (existence + source verification) --------
    # Existence gate
    eu_avail_exists = evaluator.add_custom_node(
        result=bool(_safe(item.model_name)) and bool(item.eu_availability_urls),
        id=f"{prefix_id}_eu_availability_exists",
        desc="Specific model identified and EU market availability sources provided",
        parent=device_node,
        critical=True
    )

    # Verify availability with sources
    eu_avail_leaf = evaluator.add_leaf(
        id=f"{prefix_id}_eu_market_availability_with_source",
        desc="EU market availability supported by cited sources",
        parent=device_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The device model '{_safe(item.model_name)}' is sold/available in the EU market.",
        node=eu_avail_leaf,
        sources=item.eu_availability_urls,
        additional_instruction=(
            "Verify that the cited pages indicate EU-region product availability. EU manufacturer pages or EU retailer listings suffice."
        )
    )

    # -------- Category Classification (device is the expected category) --------
    cat_leaf = evaluator.add_leaf(
        id=f"{prefix_id}_category_classification",
        desc=f"Device classified as '{expected_category}'",
        parent=device_node,
        critical=True
    )
    cat_sources = _dedup_urls(item.model_info_urls, item.eu_availability_urls, item.charging_port_urls)
    await evaluator.verify(
        claim=f"The model '{_safe(item.model_name)}' is a '{expected_category}'.",
        node=cat_leaf,
        sources=cat_sources,
        additional_instruction="Allow minor naming variants (e.g., 'mobile phone' vs 'smartphone')."
    )

    # -------- Category Coverage & Effective Date with EU source --------
    eu_cat_src_exists = evaluator.add_custom_node(
        result=bool(item.eu_category_urls),
        id=f"{prefix_id}_eu_category_sources_present",
        desc="EU regulatory source URLs provided for category coverage & effective date",
        parent=device_node,
        critical=True
    )

    eff_date_expected = _expected_deadline_for_category(expected_category)
    coverage_leaf = evaluator.add_leaf(
        id=f"{prefix_id}_category_coverage_and_effective_date_with_eu_source",
        desc=f"Category coverage & effective date ({eff_date_expected}) supported by EU source(s)",
        parent=device_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The category '{expected_category}' is covered under the EU USB-C charging port mandate, "
            f"and the applicable effective date is {eff_date_expected}."
        ),
        node=coverage_leaf,
        sources=item.eu_category_urls,
        additional_instruction=(
            "Use EU regulatory webpages to check coverage and effective dates. "
            "Most categories: December 28, 2024; laptops: April 28, 2026."
        )
    )

    # -------- Charging Port Type with source --------
    port_src_exists = evaluator.add_custom_node(
        result=bool(item.charging_port_urls),
        id=f"{prefix_id}_charging_port_sources_present",
        desc="Charging port type sources provided",
        parent=device_node,
        critical=True
    )

    port_leaf = evaluator.add_leaf(
        id=f"{prefix_id}_charging_port_type_with_source",
        desc="Charging port/connector type supported by technical source(s)",
        parent=device_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The device '{_safe(item.model_name)}' uses '{_safe(item.charging_port_type)}' as its wired charging connector type.",
        node=port_leaf,
        sources=item.charging_port_urls,
        additional_instruction="Prefer manufacturer specs; allow reputable publications. Treat 'USB Type-C' and 'USB-C' as equivalent."
    )

    # -------- Charging Power & <100W assessment with source --------
    power_src_exists = evaluator.add_custom_node(
        result=bool(item.charging_power_urls),
        id=f"{prefix_id}_charging_power_sources_present",
        desc="Charging power sources provided",
        parent=device_node,
        critical=True
    )

    # Prepare power assessment text
    watts_text = _safe(item.charging_power_watts)
    max_watts = _extract_max_watts(watts_text)
    if max_watts is not None:
        under_txt = "under" if max_watts < 100.0 else "over"
    else:
        under_txt = "unknown relative to"

    power_leaf = evaluator.add_leaf(
        id=f"{prefix_id}_charging_power_w_and_under_100w_assessment_with_source",
        desc="Charging power stated and <100W threshold assessment supported by source(s)",
        parent=device_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The device's charging power is '{watts_text}' watts, and it is {under_txt} the 100-watt threshold for mandate applicability."
        ),
        node=power_leaf,
        sources=item.charging_power_urls,
        additional_instruction=(
            "Check the stated wattage on the cited pages; if multiple values exist, consider maximum sustained charging power. "
            "Assess whether it is below 100W."
        )
    )

    # -------- Exemption checks (too small & wireless-only) with sources --------
    ex_src_exists = evaluator.add_custom_node(
        result=bool(item.exemption_urls) or bool(item.charging_port_urls),
        id=f"{prefix_id}_exemption_sources_present",
        desc="Exemption check sources provided (dimensions or charging capability)",
        parent=device_node,
        critical=True
    )

    exemption_node = evaluator.add_parallel(
        id=f"{prefix_id}_exemption_checks",
        desc="Exemption eligibility checks (size & wireless-only) with sources",
        parent=device_node,
        critical=False
    )

    # Too small exemption
    too_small_bool = _parse_yes_no(item.exemption_too_small)
    too_small_leaf = evaluator.add_leaf(
        id=f"{prefix_id}_exemption_too_small_with_source",
        desc="Too-small physical exemption determination supported by sources",
        parent=exemption_node,
        critical=True
    )
    if too_small_bool is True:
        too_small_claim = (
            f"The device '{_safe(item.model_name)}' is too small to physically accommodate a USB-C port."
        )
    elif too_small_bool is False:
        too_small_claim = (
            f"The device '{_safe(item.model_name)}' is not too small to physically accommodate a USB-C port."
        )
    else:
        too_small_claim = (
            f"No definitive determination is provided regarding whether '{_safe(item.model_name)}' is too small to accommodate USB-C."
        )
    await evaluator.verify(
        claim=too_small_claim,
        node=too_small_leaf,
        sources=_dedup_urls(item.exemption_urls, item.model_info_urls),
        additional_instruction="Use dimensions or connector layout evidence to judge physical feasibility."
    )

    # Wireless-only exemption
    wireless_only_bool = _parse_yes_no(item.exemption_wireless_only)
    wireless_leaf = evaluator.add_leaf(
        id=f"{prefix_id}_exemption_wireless_only_with_source",
        desc="Wireless-only charging exemption determination supported by sources",
        parent=exemption_node,
        critical=True
    )
    if wireless_only_bool is True:
        wireless_claim = (
            f"The device '{_safe(item.model_name)}' charges exclusively through wireless means and has no wired charging capability."
        )
        wireless_sources = _dedup_urls(item.exemption_urls, item.model_info_urls)
    elif wireless_only_bool is False:
        wireless_claim = (
            f"The device '{_safe(item.model_name)}' is not wireless-only; it supports wired charging."
        )
        # Use charging port sources to show wired capability
        wireless_sources = _dedup_urls(item.exemption_urls, item.charging_port_urls, item.model_info_urls)
    else:
        wireless_claim = (
            f"No definitive determination is provided whether '{_safe(item.model_name)}' charges exclusively via wireless means."
        )
        wireless_sources = _dedup_urls(item.exemption_urls, item.model_info_urls)
    await evaluator.verify(
        claim=wireless_claim,
        node=wireless_leaf,
        sources=wireless_sources,
        additional_instruction="Use charging capability specs to verify whether the device is wireless-only or has wired charging."
    )

    # -------- USB PD support (non-critical) --------
    usbpd_src_exists = evaluator.add_custom_node(
        result=bool(item.usb_pd_urls),
        id=f"{prefix_id}_usb_pd_sources_present",
        desc="USB PD support sources provided",
        parent=device_node,
        critical=False
    )

    usbpd_bool = _parse_yes_no(item.usb_pd_support)
    usbpd_leaf = evaluator.add_leaf(
        id=f"{prefix_id}_usb_pd_support_with_source",
        desc="USB PD support status supported by source(s)",
        parent=device_node,
        critical=False
    )
    if usbpd_bool is True:
        usbpd_claim = f"The device '{_safe(item.model_name)}' supports USB Power Delivery (USB PD)."
    elif usbpd_bool is False:
        usbpd_claim = f"The device '{_safe(item.model_name)}' does not support USB Power Delivery (USB PD)."
    else:
        usbpd_claim = f"USB Power Delivery (USB PD) support status for '{_safe(item.model_name)}' is not determined."
    await evaluator.verify(
        claim=usbpd_claim,
        node=usbpd_leaf,
        sources=_dedup_urls(item.usb_pd_urls, item.charging_port_urls),
        additional_instruction="Check specs for USB PD support; PD profiles or charger compatibility count."
    )

    # -------- Overall compliance status (critical; logical consistency) --------
    compliance_leaf = evaluator.add_leaf(
        id=f"{prefix_id}_overall_compliance_status_as_of_{AS_OF_DATE.replace(',', '').replace(' ', '_').lower()}",
        desc=f"Overall compliance status as of {AS_OF_DATE}",
        parent=device_node,
        critical=True
    )

    # Build a logical consistency claim summarizing key factors.
    # Use expected deadline from category mapping.
    expected_deadline = eff_date_expected
    meets_bool = _parse_yes_no(item.compliance_meets_requirement)
    meets_text = "meets" if meets_bool is True else ("does not meet" if meets_bool is False else "has unknown compliance relative to")

    # Compose derived under/over string for clarity in final claim as well
    under_over_text = ""
    if max_watts is not None:
        under_over_text = "under 100W" if max_watts < 100.0 else "at/over 100W"
    else:
        under_over_text = "unknown relative to 100W"

    compliance_claim = (
        f"As of {AS_OF_DATE}, the applicable compliance deadline for the '{expected_category}' category is {expected_deadline}. "
        f"The device '{_safe(item.model_name)}' has charging port type '{_safe(item.charging_port_type)}', charging power '{_safe(item.charging_power_watts)}' ({under_over_text}), "
        f"exemption-too-small='{_safe(item.exemption_too_small)}', exemption-wireless-only='{_safe(item.exemption_wireless_only)}'. "
        f"Given the EU deadlines (most categories Dec 28, 2024; laptops Apr 28, 2026) and exemption criteria, the stated overall status "
        f"('{_safe(item.compliance_status)}') and determination that it {meets_text} the EU USB-C requirement are logically consistent."
    )

    # Ensure compliance determination depends on prior critical checks to avoid premature passing
    await evaluator.verify(
        claim=compliance_claim,
        node=compliance_leaf,
        sources=None,
        extra_prerequisites=[
            eu_avail_leaf,           # EU availability confirmed
            cat_leaf,                # Device category classification
            coverage_leaf,           # Category coverage & effective date
            port_leaf,               # Port type
            power_leaf,              # Power assessment
            too_small_leaf,          # Exemption too small
            wireless_leaf            # Exemption wireless-only
        ],
        additional_instruction=(
            "Judge internal logical consistency using the extracted facts and the EU deadlines mapping: "
            "December 28, 2024 for smartphone, wireless earbuds, handheld gaming console; April 28, 2026 for laptops. "
            "If any required upstream checks failed or are unsupported, mark this compliance determination as inconsistent."
        )
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the EU USB-C compliance analysis task.
    """
    # Initialize evaluator (root parallel aggregation to treat each device independently)
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

    # Record ground truth / expected deadlines mapping
    evaluator.add_ground_truth({
        "as_of_date": AS_OF_DATE,
        "expected_mandate_deadlines": MANDATE_DEADLINES
    }, gt_type="expected_deadlines")

    # Extract device information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_devices(),
        template_class=DevicesExtraction,
        extraction_name="devices_extraction"
    )

    # Build verification tree per device category
    # Smartphone
    smartphone_item = extracted.smartphone or DeviceComplianceItem()
    await verify_device(
        evaluator=evaluator,
        parent_node=root,
        item=smartphone_item,
        prefix_id="smartphone",
        device_desc="One smartphone device compliance analysis.",
        expected_category="smartphone"
    )

    # Laptop
    laptop_item = extracted.laptop or DeviceComplianceItem()
    await verify_device(
        evaluator=evaluator,
        parent_node=root,
        item=laptop_item,
        prefix_id="laptop",
        device_desc="One laptop device compliance analysis.",
        expected_category="laptop"
    )

    # Wireless earbuds
    earbuds_item = extracted.wireless_earbuds or DeviceComplianceItem()
    await verify_device(
        evaluator=evaluator,
        parent_node=root,
        item=earbuds_item,
        prefix_id="wireless_earbuds",
        device_desc="One wireless earbuds device compliance analysis.",
        expected_category="wireless earbuds"
    )

    # Handheld gaming console
    console_item = extracted.handheld_gaming_console or DeviceComplianceItem()
    await verify_device(
        evaluator=evaluator,
        parent_node=root,
        item=console_item,
        prefix_id="handheld_gaming_console",
        device_desc="One handheld gaming console device compliance analysis.",
        expected_category="handheld gaming console"
    )

    # Return final structured summary
    return evaluator.get_summary()