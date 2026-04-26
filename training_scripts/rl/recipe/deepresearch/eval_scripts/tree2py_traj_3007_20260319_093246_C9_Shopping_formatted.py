import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "retail_membership_2026_comparison"
TASK_DESCRIPTION = """You are evaluating retail membership programs for 2026. Provide comprehensive comparison information for three major US retail memberships: Target Circle 360, Walmart Plus, and Amazon Prime.

For each membership program, provide the following details with supporting reference URLs from official sources:

Pricing Category:
- Standard annual membership cost
- Standard monthly membership cost
- Special discount pricing for college students or government assistance recipients (including the discounted rates)

Delivery Category:
- Minimum order threshold required for free same-day delivery
- Available delivery speed options (same-day, next-day, two-day)
- Return period extensions or special return benefits

Payment & Savings Category:
- Name of the associated credit or debit card that provides rewards
- Cashback percentage rate earned at the retailer with the associated card
- Fuel savings program details (cents per gallon discount and participating gas station brands)

Additional Perks Category:
- Included streaming service(s)
- Pharmacy delivery benefits or prescription services
- Photo storage benefits or monthly special member offers

Your response should include specific factual values for each data point and provide reference URLs to verify the information.
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class DataPoint(BaseModel):
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ListDataPoint(BaseModel):
    items: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class PricingInfo(BaseModel):
    annual_cost: Optional[DataPoint] = None
    monthly_cost: Optional[DataPoint] = None
    discount_pricing: Optional[DataPoint] = None


class DeliveryInfo(BaseModel):
    free_same_day_threshold: Optional[DataPoint] = None
    delivery_speed_options: Optional[ListDataPoint] = None
    return_benefits: Optional[DataPoint] = None


class PaymentSavingsInfo(BaseModel):
    associated_card: Optional[DataPoint] = None
    cashback_rate: Optional[DataPoint] = None
    fuel_savings: Optional[DataPoint] = None  # Accepts "none" or descriptive details


class AdditionalPerksInfo(BaseModel):
    streaming: Optional[DataPoint] = None  # Accepts "none" or descriptive details
    pharmacy: Optional[DataPoint] = None   # Accepts "none" or descriptive details
    photo_storage: Optional[DataPoint] = None
    monthly_offers: Optional[DataPoint] = None  # e.g., "monthly freebies on 1st"


class ProgramInfo(BaseModel):
    pricing: Optional[PricingInfo] = None
    delivery: Optional[DeliveryInfo] = None
    payment_and_savings: Optional[PaymentSavingsInfo] = None
    additional_perks: Optional[AdditionalPerksInfo] = None


class MembershipExtraction(BaseModel):
    target_circle_360: Optional[ProgramInfo] = None
    walmart_plus: Optional[ProgramInfo] = None
    amazon_prime: Optional[ProgramInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_memberships() -> str:
    return """
Extract structured data for three membership programs (Target Circle 360, Walmart+, Amazon Prime) strictly from the provided answer. For every datapoint, also extract the official-source URL(s) cited in the answer that supports that datapoint. If a datapoint is not stated in the answer, set it to null (or empty list for lists). Do not invent values or URLs.

Return JSON with this schema:

{
  "target_circle_360": {
    "pricing": {
      "annual_cost": { "value": string|null, "urls": string[] },
      "monthly_cost": { "value": string|null, "urls": string[] },
      "discount_pricing": { "value": string|null, "urls": string[] }
    },
    "delivery": {
      "free_same_day_threshold": { "value": string|null, "urls": string[] },
      "delivery_speed_options": { "items": string[], "urls": string[] },
      "return_benefits": { "value": string|null, "urls": string[] }
    },
    "payment_and_savings": {
      "associated_card": { "value": string|null, "urls": string[] },
      "cashback_rate": { "value": string|null, "urls": string[] },
      "fuel_savings": { "value": string|null, "urls": string[] }  // may be "none"
    },
    "additional_perks": {
      "streaming": { "value": string|null, "urls": string[] },   // may be "none"
      "pharmacy": { "value": string|null, "urls": string[] },    // may be "none"
      "photo_storage": { "value": string|null, "urls": string[] },
      "monthly_offers": { "value": string|null, "urls": string[] }
    }
  },
  "walmart_plus": {
    "pricing": {
      "annual_cost": { "value": string|null, "urls": string[] },
      "monthly_cost": { "value": string|null, "urls": string[] },
      "discount_pricing": { "value": string|null, "urls": string[] }
    },
    "delivery": {
      "free_same_day_threshold": { "value": string|null, "urls": string[] },
      "delivery_speed_options": { "items": string[], "urls": string[] },
      "return_benefits": { "value": string|null, "urls": string[] }
    },
    "payment_and_savings": {
      "associated_card": { "value": string|null, "urls": string[] },
      "cashback_rate": { "value": string|null, "urls": string[] },
      "fuel_savings": { "value": string|null, "urls": string[] }
    },
    "additional_perks": {
      "streaming": { "value": string|null, "urls": string[] },
      "pharmacy": { "value": string|null, "urls": string[] },
      "photo_storage": { "value": string|null, "urls": string[] },
      "monthly_offers": { "value": string|null, "urls": string[] }
    }
  },
  "amazon_prime": {
    "pricing": {
      "annual_cost": { "value": string|null, "urls": string[] },
      "monthly_cost": { "value": string|null, "urls": string[] },
      "discount_pricing": { "value": string|null, "urls": string[] }
    },
    "delivery": {
      "free_same_day_threshold": { "value": string|null, "urls": string[] },
      "delivery_speed_options": { "items": string[], "urls": string[] },
      "return_benefits": { "value": string|null, "urls": string[] }
    },
    "payment_and_savings": {
      "associated_card": { "value": string|null, "urls": string[] },
      "cashback_rate": { "value": string|null, "urls": string[] },
      "fuel_savings": { "value": string|null, "urls": string[] }
    },
    "additional_perks": {
      "streaming": { "value": string|null, "urls": string[] },
      "pharmacy": { "value": string|null, "urls": string[] },
      "photo_storage": { "value": string|null, "urls": string[] },
      "monthly_offers": { "value": string|null, "urls": string[] }
    }
  }
}

Notes:
- Use exact strings and URLs as they appear in the answer.
- For list fields like delivery_speed_options.items, include each stated speed (e.g., "same-day", "next-day", "two-day"). Keep them as simple lowercase strings if possible.
- URLs must be actual links from the answer text; ignore non-URL mentions of sources.
"""


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _is_official(url: str, allowed_domains: List[str]) -> bool:
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower()
        # Strip leading "www."
        if host.startswith("www."):
            host = host[4:]
        return any(host == d or host.endswith("." + d) for d in allowed_domains)
    except Exception:
        return False


def _has_value(dp: Optional[DataPoint]) -> bool:
    return bool(dp and dp.value and str(dp.value).strip())


def _has_sources(dp_or_listdp: Any) -> bool:
    try:
        urls = getattr(dp_or_listdp, "urls", [])
        return bool(urls and len(urls) > 0)
    except Exception:
        return False


def _has_official_source(dp_or_listdp: Any, allowed_domains: List[str]) -> bool:
    try:
        urls = getattr(dp_or_listdp, "urls", [])
        return any(_is_official(u, allowed_domains) for u in urls)
    except Exception:
        return False


def _normalize_speed_token(s: str) -> str:
    t = s.strip().lower().replace(" ", "").replace("-", "")
    if "sameday" in t:
        return "same-day"
    if "nextday" in t or "oneday" in t or t in {"1day", "1d"}:
        return "next-day"
    if "twoday" in t or t in {"2day", "2d"}:
        return "two-day"
    # fall back to original readable
    return s.strip().lower()


def _contains_required_speeds(items: List[str], required: List[str]) -> bool:
    norm_items = {_normalize_speed_token(x) for x in items}
    norm_required = {_normalize_speed_token(x) for x in required}
    return norm_required.issubset(norm_items)


def _program_domains(program_key: str, field_key: Optional[str] = None) -> List[str]:
    """
    Whitelist of official domains per program, with some field-specific allowances.
    """
    if program_key == "target":
        # Target-owned and core partner for delivery (Shipt is a Target company)
        return ["target.com", "shipt.com"]
    if program_key == "walmart":
        # Walmart-owned + likely card issuer domains (transitioning from Capital One to One)
        if field_key in {"associated_card", "cashback_rate"}:
            return ["walmart.com", "one.app", "capitalone.com"]
        return ["walmart.com"]
    if program_key == "amazon":
        # Amazon and official issuer for Prime Visa
        if field_key in {"associated_card", "cashback_rate"}:
            return ["amazon.com", "chase.com"]
        # For other benefits, amazon.com (and sub-brands) is sufficient
        return ["amazon.com", "aboutamazon.com", "wholefoodsmarket.com", "primevideo.com"]
    return []


# --------------------------------------------------------------------------- #
# Verification building blocks                                                #
# --------------------------------------------------------------------------- #
async def _verify_simple_datapoint(
    evaluator: Evaluator,
    parent,
    *,
    node_base_id: str,
    program_key: str,
    field_key: str,
    dp: Optional[DataPoint],
    claim_text: str,
    field_desc_for_nodes: str,
    add_ins: str = "None",
    critical: bool = True,
) -> None:
    """
    Build a parallel group for a simple DataPoint:
      - value exists (critical)
      - sources exist (critical)
      - official-source present (critical)
      - verify claim against URLs (critical)
    """
    group = evaluator.add_parallel(
        id=f"{node_base_id}_group",
        desc=field_desc_for_nodes,
        parent=parent,
        critical=critical,
    )

    # Existence checks
    evaluator.add_custom_node(
        result=_has_value(dp),
        id=f"{node_base_id}_has_value",
        desc=f"{field_desc_for_nodes} - value provided in the answer",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_sources(dp),
        id=f"{node_base_id}_has_sources",
        desc=f"{field_desc_for_nodes} - at least one reference URL provided",
        parent=group,
        critical=True
    )

    # Official-source check
    allowed = _program_domains(program_key, field_key)
    evaluator.add_custom_node(
        result=_has_official_source(dp, allowed),
        id=f"{node_base_id}_official_source",
        desc=f"{field_desc_for_nodes} - at least one official-source URL provided",
        parent=group,
        critical=True
    )

    # Verification leaf
    leaf = evaluator.add_leaf(
        id=f"{node_base_id}_supported",
        desc=field_desc_for_nodes,
        parent=group,
        critical=True
    )
    urls = dp.urls if dp and dp.urls else []
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=urls,
        additional_instruction=add_ins
    )


async def _verify_list_datapoint_with_required_includes(
    evaluator: Evaluator,
    parent,
    *,
    node_base_id: str,
    program_key: str,
    field_key: str,
    ldp: Optional[ListDataPoint],
    required_includes: Optional[List[str]],
    claim_prefix: str,
    field_desc_for_nodes: str,
    add_ins: str = "None",
    critical: bool = True,
) -> None:
    """
    Build a parallel group for a ListDataPoint with optional "must include" subset.
    """
    group = evaluator.add_parallel(
        id=f"{node_base_id}_group",
        desc=field_desc_for_nodes,
        parent=parent,
        critical=critical
    )

    # Existence checks
    items_exist = bool(ldp and isinstance(ldp.items, list) and len(ldp.items) > 0)
    evaluator.add_custom_node(
        result=items_exist,
        id=f"{node_base_id}_has_items",
        desc=f"{field_desc_for_nodes} - list of items is provided in the answer",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_sources(ldp),
        id=f"{node_base_id}_has_sources",
        desc=f"{field_desc_for_nodes} - at least one reference URL provided",
        parent=group,
        critical=True
    )

    # Official-source check
    allowed = _program_domains(program_key, field_key)
    evaluator.add_custom_node(
        result=_has_official_source(ldp, allowed),
        id=f"{node_base_id}_official_source",
        desc=f"{field_desc_for_nodes} - at least one official-source URL provided",
        parent=group,
        critical=True
    )

    # Required includes (if any)
    if required_includes:
        includes_ok = items_exist and _contains_required_speeds(ldp.items, required_includes)
        evaluator.add_custom_node(
            result=includes_ok,
            id=f"{node_base_id}_includes_required",
            desc=f"{field_desc_for_nodes} - includes required items: {', '.join(required_includes)} (allowing reasonable synonyms)",
            parent=group,
            critical=True
        )

    # Verification leaf
    items_text = ", ".join(ldp.items) if ldp and ldp.items else ""
    claim = f"{claim_prefix} {items_text}".strip()
    leaf = evaluator.add_leaf(
        id=f"{node_base_id}_supported",
        desc=field_desc_for_nodes,
        parent=group,
        critical=True
    )
    urls = ldp.urls if ldp and ldp.urls else []
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Program-specific verification                                               #
# --------------------------------------------------------------------------- #
async def _verify_target_circle_360(evaluator: Evaluator, root: Any, info: Optional[ProgramInfo]) -> None:
    prog_node = evaluator.add_parallel(
        id="program_target_circle_360",
        desc="Target Circle 360 datapoints (with official-source URLs supporting each datapoint).",
        parent=root,
        critical=False
    )

    pricing = evaluator.add_parallel(
        id="tc360_pricing",
        desc="Target Circle 360 pricing datapoints.",
        parent=prog_node,
        critical=True
    )
    delivery = evaluator.add_parallel(
        id="tc360_delivery",
        desc="Target Circle 360 delivery datapoints.",
        parent=prog_node,
        critical=True
    )
    pay_save = evaluator.add_parallel(
        id="tc360_payment_and_savings",
        desc="Target Circle 360 payment & savings datapoints.",
        parent=prog_node,
        critical=True
    )
    perks = evaluator.add_parallel(
        id="tc360_additional_perks",
        desc="Target Circle 360 additional perks datapoints.",
        parent=prog_node,
        critical=True
    )

    # Pricing
    p = info.pricing if info else PricingInfo()
    await _verify_simple_datapoint(
        evaluator, pricing,
        node_base_id="tc360_pricing_annual_cost",
        program_key="target",
        field_key="annual_cost",
        dp=p.annual_cost,
        claim_text=f"The standard annual membership cost for Target Circle 360 is {p.annual_cost.value} per year." if _has_value(p.annual_cost) else "",
        field_desc_for_nodes="States standard annual membership cost AND provides an official-source URL supporting it.",
        add_ins="Verify standard (non-promotional) annual pricing as stated on Target's official site. Ignore limited-time offers."
    )
    await _verify_simple_datapoint(
        evaluator, pricing,
        node_base_id="tc360_pricing_monthly_cost",
        program_key="target",
        field_key="monthly_cost",
        dp=p.monthly_cost,
        claim_text=f"The standard monthly membership cost for Target Circle 360 is {p.monthly_cost.value} per month." if _has_value(p.monthly_cost) else "",
        field_desc_for_nodes="States standard monthly membership cost AND provides an official-source URL supporting it.",
        add_ins="Verify standard (non-promotional) monthly pricing as stated on Target's official site."
    )
    await _verify_simple_datapoint(
        evaluator, pricing,
        node_base_id="tc360_pricing_discount",
        program_key="target",
        field_key="discount_pricing",
        dp=p.discount_pricing,
        claim_text=f"Target Circle 360 offers discounted pricing for eligible students or government assistance recipients: {p.discount_pricing.value}." if _has_value(p.discount_pricing) else "",
        field_desc_for_nodes="States special discount pricing for students/assistance recipients (including discounted rates) AND provides an official-source URL supporting it.",
        add_ins="Confirm discounted rates and eligibility language on Target's official site; accept phrasing like '$49/year with Target Circle Card' and '$4.99/month' if shown."
    )

    # Delivery
    d = info.delivery if info else DeliveryInfo()
    await _verify_simple_datapoint(
        evaluator, delivery,
        node_base_id="tc360_delivery_threshold",
        program_key="target",
        field_key="free_same_day_threshold",
        dp=d.free_same_day_threshold,
        claim_text=f"The minimum order threshold required for free same-day delivery with Target Circle 360 is {d.free_same_day_threshold.value}." if _has_value(d.free_same_day_threshold) else "",
        field_desc_for_nodes="States minimum order threshold required for free same-day delivery AND provides an official-source URL supporting it.",
        add_ins="Confirm threshold applies to same-day delivery benefit for members."
    )
    await _verify_list_datapoint_with_required_includes(
        evaluator, delivery,
        node_base_id="tc360_delivery_speeds",
        program_key="target",
        field_key="delivery_speed_options",
        ldp=d.delivery_speed_options,
        required_includes=["same-day", "next-day", "two-day"],
        claim_prefix="The stated delivery speed options for Target Circle 360 include:",
        field_desc_for_nodes="States delivery speed options include same-day, next-day, and two-day AND provides an official-source URL supporting it.",
        add_ins="Allow reasonable synonyms (e.g., '1-day' for next-day, '2-day' for two-day)."
    )
    await _verify_simple_datapoint(
        evaluator, delivery,
        node_base_id="tc360_delivery_returns",
        program_key="target",
        field_key="return_benefits",
        dp=d.return_benefits,
        claim_text=f"Target Circle 360 members receive the following return benefits: {d.return_benefits.value}." if _has_value(d.return_benefits) else "",
        field_desc_for_nodes="States return benefit and provides an official-source URL supporting it.",
        add_ins="If the answer claims '30 additional days beyond standard policy', verify that exact extension on official Target pages."
    )

    # Payment & Savings
    ps = info.payment_and_savings if info else PaymentSavingsInfo()
    await _verify_simple_datapoint(
        evaluator, pay_save,
        node_base_id="tc360_pay_card",
        program_key="target",
        field_key="associated_card",
        dp=ps.associated_card,
        claim_text=f"The associated rewards card for Target is {ps.associated_card.value}." if _has_value(ps.associated_card) else "",
        field_desc_for_nodes="Names associated rewards card AND provides an official-source URL supporting it.",
        add_ins="Confirm the official card name on Target's site."
    )
    await _verify_simple_datapoint(
        evaluator, pay_save,
        node_base_id="tc360_pay_cashback",
        program_key="target",
        field_key="cashback_rate",
        dp=ps.cashback_rate,
        claim_text=f"The cashback rate earned at Target with the associated card is {ps.cashback_rate.value}." if _has_value(ps.cashback_rate) else "",
        field_desc_for_nodes="States cashback rate at Target with the associated card AND provides an official-source URL supporting it.",
        add_ins="Verify numeric cashback percentage on official Target pages."
    )
    await _verify_simple_datapoint(
        evaluator, pay_save,
        node_base_id="tc360_pay_fuel",
        program_key="target",
        field_key="fuel_savings",
        dp=ps.fuel_savings,
        claim_text=f"Fuel savings benefit for Target Circle 360: {ps.fuel_savings.value}." if _has_value(ps.fuel_savings) else "",
        field_desc_for_nodes="Provides fuel savings details or explicitly states none, AND provides an official-source URL supporting the claim.",
        add_ins="If the claim is that no fuel savings exist, verify that the official pages do not list a fuel discount benefit."
    )

    # Additional Perks
    ap = info.additional_perks if info else AdditionalPerksInfo()
    await _verify_simple_datapoint(
        evaluator, perks,
        node_base_id="tc360_perks_streaming",
        program_key="target",
        field_key="streaming",
        dp=ap.streaming,
        claim_text=f"Included streaming service(s) for Target Circle 360: {ap.streaming.value}." if _has_value(ap.streaming) else "",
        field_desc_for_nodes="States included streaming service(s) (or none) AND provides an official-source URL supporting it.",
        add_ins="Mark unsupported if the page is not an official Target source."
    )
    await _verify_simple_datapoint(
        evaluator, perks,
        node_base_id="tc360_perks_pharmacy",
        program_key="target",
        field_key="pharmacy",
        dp=ap.pharmacy,
        claim_text=f"Pharmacy delivery/prescription benefits for Target Circle 360: {ap.pharmacy.value}." if _has_value(ap.pharmacy) else "",
        field_desc_for_nodes="States pharmacy delivery benefits or prescription services (or none) AND provides an official-source URL supporting it.",
        add_ins="Only accept if described on official Target or Target-owned pages."
    )
    await _verify_simple_datapoint(
        evaluator, perks,
        node_base_id="tc360_perks_monthly_offers",
        program_key="target",
        field_key="monthly_offers",
        dp=ap.monthly_offers,
        claim_text=f"Monthly special member offers: {ap.monthly_offers.value}." if _has_value(ap.monthly_offers) else "",
        field_desc_for_nodes="States monthly special member offers (e.g., monthly freebies on the first) AND provides an official-source URL supporting it.",
        add_ins="If the claim mentions 'monthly freebies on the first of each month', verify that phrasing (or equivalent) on official Target pages."
    )


async def _verify_walmart_plus(evaluator: Evaluator, root: Any, info: Optional[ProgramInfo]) -> None:
    prog_node = evaluator.add_parallel(
        id="program_walmart_plus",
        desc="Walmart+ datapoints (with official-source URLs supporting each datapoint).",
        parent=root,
        critical=False
    )

    pricing = evaluator.add_parallel(
        id="wm_pricing",
        desc="Walmart+ pricing datapoints.",
        parent=prog_node,
        critical=True
    )
    delivery = evaluator.add_parallel(
        id="wm_delivery",
        desc="Walmart+ delivery datapoints.",
        parent=prog_node,
        critical=True
    )
    pay_save = evaluator.add_parallel(
        id="wm_payment_and_savings",
        desc="Walmart+ payment & savings datapoints.",
        parent=prog_node,
        critical=True
    )
    perks = evaluator.add_parallel(
        id="wm_additional_perks",
        desc="Walmart+ additional perks datapoints.",
        parent=prog_node,
        critical=True
    )

    # Pricing
    p = info.pricing if info else PricingInfo()
    await _verify_simple_datapoint(
        evaluator, pricing,
        node_base_id="wm_pricing_annual_cost",
        program_key="walmart",
        field_key="annual_cost",
        dp=p.annual_cost,
        claim_text=f"The standard annual membership cost for Walmart+ is {p.annual_cost.value} per year." if _has_value(p.annual_cost) else "",
        field_desc_for_nodes="States standard annual membership cost AND provides an official-source URL supporting it.",
        add_ins="Verify standard (non-promotional) annual pricing on Walmart's official site."
    )
    await _verify_simple_datapoint(
        evaluator, pricing,
        node_base_id="wm_pricing_monthly_cost",
        program_key="walmart",
        field_key="monthly_cost",
        dp=p.monthly_cost,
        claim_text=f"The standard monthly membership cost for Walmart+ is {p.monthly_cost.value} per month." if _has_value(p.monthly_cost) else "",
        field_desc_for_nodes="States standard monthly membership cost AND provides an official-source URL supporting it.",
        add_ins="Verify standard (non-promotional) monthly pricing on Walmart's official site."
    )
    await _verify_simple_datapoint(
        evaluator, pricing,
        node_base_id="wm_pricing_discount",
        program_key="walmart",
        field_key="discount_pricing",
        dp=p.discount_pricing,
        claim_text=f"Walmart+ offers discounted pricing for eligible students or government assistance recipients: {p.discount_pricing.value}." if _has_value(p.discount_pricing) else "",
        field_desc_for_nodes="States special discount pricing for students/assistance recipients (including discounted rates) AND provides an official-source URL supporting it.",
        add_ins="Confirm discounted rates and eligibility language on Walmart's official site."
    )

    # Delivery
    d = info.delivery if info else DeliveryInfo()
    await _verify_simple_datapoint(
        evaluator, delivery,
        node_base_id="wm_delivery_threshold",
        program_key="walmart",
        field_key="free_same_day_threshold",
        dp=d.free_same_day_threshold,
        claim_text=f"The minimum order threshold required for free delivery with Walmart+ is {d.free_same_day_threshold.value}." if _has_value(d.free_same_day_threshold) else "",
        field_desc_for_nodes="States minimum order threshold required for free delivery AND provides an official-source URL supporting it.",
        add_ins="Confirm that threshold applies to same-day/express where stated; otherwise shipping threshold for Walmart+."
    )
    await _verify_list_datapoint_with_required_includes(
        evaluator, delivery,
        node_base_id="wm_delivery_speeds",
        program_key="walmart",
        field_key="delivery_speed_options",
        ldp=d.delivery_speed_options,
        required_includes=["same-day", "next-day", "two-day"],
        claim_prefix="The stated delivery speed options for Walmart+ include:",
        field_desc_for_nodes="States delivery speed options include same-day, next-day, and two-day AND provides an official-source URL supporting it.",
        add_ins="Allow synonyms (e.g., 'NextDay' or '1-day' for next-day)."
    )
    await _verify_simple_datapoint(
        evaluator, delivery,
        node_base_id="wm_delivery_returns",
        program_key="walmart",
        field_key="return_benefits",
        dp=d.return_benefits,
        claim_text=f"Walmart+ return benefits: {d.return_benefits.value}." if _has_value(d.return_benefits) else "",
        field_desc_for_nodes="Describes return period extensions or special return benefits (or none) AND provides an official-source URL supporting it.",
        add_ins="Accept 'none' only if official Walmart pages do not list special return benefits."
    )

    # Payment & Savings
    ps = info.payment_and_savings if info else PaymentSavingsInfo()
    await _verify_simple_datapoint(
        evaluator, pay_save,
        node_base_id="wm_pay_card",
        program_key="walmart",
        field_key="associated_card",
        dp=ps.associated_card,
        claim_text=f"The associated rewards card for Walmart is {ps.associated_card.value}." if _has_value(ps.associated_card) else "",
        field_desc_for_nodes="Names associated rewards card AND provides an official-source URL (Walmart and/or official issuer/partner) supporting it.",
        add_ins="Issuer/partner page (e.g., One or Capital One) also counts as official."
    )
    await _verify_simple_datapoint(
        evaluator, pay_save,
        node_base_id="wm_pay_cashback",
        program_key="walmart",
        field_key="cashback_rate",
        dp=ps.cashback_rate,
        claim_text=f"The cashback rate earned at Walmart with the associated card is {ps.cashback_rate.value}." if _has_value(ps.cashback_rate) else "",
        field_desc_for_nodes="States cashback rate at Walmart with the associated card AND provides an official-source URL supporting it.",
        add_ins="Verify numeric cashback rate on Walmart or issuer/partner official pages."
    )
    await _verify_simple_datapoint(
        evaluator, pay_save,
        node_base_id="wm_pay_fuel",
        program_key="walmart",
        field_key="fuel_savings",
        dp=ps.fuel_savings,
        claim_text=f"Fuel savings for Walmart+ members: {ps.fuel_savings.value}." if _has_value(ps.fuel_savings) else "",
        field_desc_for_nodes="States fuel savings (e.g., 10¢/gal) and participating brands AND provides an official-source URL supporting it.",
        add_ins="Verify cents-per-gallon and participating brands (e.g., Walmart, Exxon, Mobil, Murphy; Sam's Club member prices) on Walmart official pages."
    )

    # Additional Perks
    ap = info.additional_perks if info else AdditionalPerksInfo()
    await _verify_simple_datapoint(
        evaluator, perks,
        node_base_id="wm_perks_streaming",
        program_key="walmart",
        field_key="streaming",
        dp=ap.streaming,
        claim_text=f"Included streaming benefit for Walmart+: {ap.streaming.value}." if _has_value(ap.streaming) else "",
        field_desc_for_nodes="States included streaming benefit (e.g., choice of Paramount+ Essential or Peacock Premium, rotatable every 90 days) AND provides an official-source URL supporting it.",
        add_ins="Verify rotation/choice details on Walmart's official pages."
    )
    await _verify_simple_datapoint(
        evaluator, perks,
        node_base_id="wm_perks_pharmacy",
        program_key="walmart",
        field_key="pharmacy",
        dp=ap.pharmacy,
        claim_text=f"Walmart+ pharmacy benefit: {ap.pharmacy.value}." if _has_value(ap.pharmacy) else "",
        field_desc_for_nodes="States free pharmacy delivery with no order minimum (or stated benefit) AND provides an official-source URL supporting it.",
        add_ins="Confirm delivery terms on official Walmart pharmacy/Walmart+ pages."
    )

    # photo_storage_or_monthly_offers: accept either one present
    # Choose primary field to verify
    primary_dp = ap.photo_storage if _has_value(ap.photo_storage) else (ap.monthly_offers if _has_value(ap.monthly_offers) else None)
    group = evaluator.add_parallel(
        id="wm_perks_photo_or_monthly_group",
        desc="Describes photo storage benefits OR monthly special member offers AND provides an official-source URL supporting it.",
        parent=perks,
        critical=True
    )
    evaluator.add_custom_node(
        result=primary_dp is not None,
        id="wm_perks_photo_or_monthly_has_either",
        desc="Either photo storage benefit or monthly special offers is provided in the answer",
        parent=group,
        critical=True
    )
    # If none present, still create leaf that will be skipped due to failed precondition
    # Official-source check for whichever chosen
    evaluator.add_custom_node(
        result=_has_sources(primary_dp) if primary_dp else False,
        id="wm_perks_photo_or_monthly_has_sources",
        desc="At least one reference URL provided for the chosen benefit",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_official_source(primary_dp, _program_domains("walmart", None)) if primary_dp else False,
        id="wm_perks_photo_or_monthly_official",
        desc="At least one official-source URL provided for the chosen benefit",
        parent=group,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="wm_perks_photo_or_monthly_supported",
        desc="Photo storage benefits OR monthly special member offers are supported by official source",
        parent=group,
        critical=True
    )
    claim = ""
    urls = []
    if primary_dp:
        claim = f"Walmart+ additional perk: {primary_dp.value}."
        urls = primary_dp.urls if primary_dp.urls else []
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="Verify this perk on Walmart's official pages."
    )


async def _verify_amazon_prime(evaluator: Evaluator, root: Any, info: Optional[ProgramInfo]) -> None:
    prog_node = evaluator.add_parallel(
        id="program_amazon_prime",
        desc="Amazon Prime datapoints (with official-source URLs supporting each datapoint).",
        parent=root,
        critical=False
    )

    pricing = evaluator.add_parallel(
        id="amz_pricing",
        desc="Amazon Prime pricing datapoints.",
        parent=prog_node,
        critical=True
    )
    delivery = evaluator.add_parallel(
        id="amz_delivery",
        desc="Amazon Prime delivery datapoints.",
        parent=prog_node,
        critical=True
    )
    pay_save = evaluator.add_parallel(
        id="amz_payment_and_savings",
        desc="Amazon Prime payment & savings datapoints.",
        parent=prog_node,
        critical=True
    )
    perks = evaluator.add_parallel(
        id="amz_additional_perks",
        desc="Amazon Prime additional perks datapoints.",
        parent=prog_node,
        critical=True
    )

    # Pricing
    p = info.pricing if info else PricingInfo()
    await _verify_simple_datapoint(
        evaluator, pricing,
        node_base_id="amz_pricing_annual_cost",
        program_key="amazon",
        field_key="annual_cost",
        dp=p.annual_cost,
        claim_text=f"The standard annual membership cost for Amazon Prime is {p.annual_cost.value} per year." if _has_value(p.annual_cost) else "",
        field_desc_for_nodes="States standard annual membership cost AND provides an official-source URL supporting it.",
        add_ins="Verify standard annual pricing on Amazon's official Prime pages."
    )
    await _verify_simple_datapoint(
        evaluator, pricing,
        node_base_id="amz_pricing_monthly_cost",
        program_key="amazon",
        field_key="monthly_cost",
        dp=p.monthly_cost,
        claim_text=f"The standard monthly membership cost for Amazon Prime is {p.monthly_cost.value} per month." if _has_value(p.monthly_cost) else "",
        field_desc_for_nodes="States standard monthly membership cost AND provides an official-source URL supporting it.",
        add_ins="Verify standard monthly pricing on Amazon's official Prime pages."
    )
    await _verify_simple_datapoint(
        evaluator, pricing,
        node_base_id="amz_pricing_discount",
        program_key="amazon",
        field_key="discount_pricing",
        dp=p.discount_pricing,
        claim_text=f"Amazon Prime offers discounted pricing for students or government assistance recipients: {p.discount_pricing.value}." if _has_value(p.discount_pricing) else "",
        field_desc_for_nodes="States special discount pricing (including discounted rates) AND provides an official-source URL supporting it.",
        add_ins="Confirm student/qualified government assistance pricing and eligibility details on Amazon official pages."
    )

    # Delivery
    d = info.delivery if info else DeliveryInfo()
    await _verify_simple_datapoint(
        evaluator, delivery,
        node_base_id="amz_delivery_threshold",
        program_key="amazon",
        field_key="free_same_day_threshold",
        dp=d.free_same_day_threshold,
        claim_text=f"The minimum order threshold required for free Same-Day delivery with Amazon Prime is {d.free_same_day_threshold.value}." if _has_value(d.free_same_day_threshold) else "",
        field_desc_for_nodes="States the minimum order threshold required for free same-day delivery AND provides an official-source URL supporting it.",
        add_ins="Confirm threshold for free Same-Day delivery on Amazon official pages."
    )
    await _verify_list_datapoint_with_required_includes(
        evaluator, delivery,
        node_base_id="amz_delivery_speeds",
        program_key="amazon",
        field_key="delivery_speed_options",
        ldp=d.delivery_speed_options,
        required_includes=["two-day", "one-day", "same-day"],
        claim_prefix="The stated delivery speed options for Amazon Prime include:",
        field_desc_for_nodes="States delivery speed options include Two-Day, One-Day, and Same-Day AND provides an official-source URL supporting it.",
        add_ins="Allow synonyms (e.g., 'Next-Day' for One-Day, '2-day' for Two-Day)."
    )
    await _verify_simple_datapoint(
        evaluator, delivery,
        node_base_id="amz_delivery_returns",
        program_key="amazon",
        field_key="return_benefits",
        dp=d.return_benefits,
        claim_text=f"Amazon Prime return benefits: {d.return_benefits.value}." if _has_value(d.return_benefits) else "",
        field_desc_for_nodes="Describes return period extensions or special return benefits (or none) AND provides an official-source URL supporting it.",
        add_ins="Accept 'none' only if Amazon official pages do not list special return benefits."
    )

    # Payment & Savings
    ps = info.payment_and_savings if info else PaymentSavingsInfo()
    await _verify_simple_datapoint(
        evaluator, pay_save,
        node_base_id="amz_pay_card",
        program_key="amazon",
        field_key="associated_card",
        dp=ps.associated_card,
        claim_text=f"The associated rewards card for Amazon Prime is {ps.associated_card.value}." if _has_value(ps.associated_card) else "",
        field_desc_for_nodes="Names associated rewards card AND provides an official-source URL (Amazon and/or official issuer) supporting it.",
        add_ins="Issuer page (e.g., Chase) also counts as official."
    )
    await _verify_simple_datapoint(
        evaluator, pay_save,
        node_base_id="amz_pay_cashback",
        program_key="amazon",
        field_key="cashback_rate",
        dp=ps.cashback_rate,
        claim_text=f"The cashback rate with the associated card at Amazon.com, Amazon Fresh, and Whole Foods Market is {ps.cashback_rate.value}." if _has_value(ps.cashback_rate) else "",
        field_desc_for_nodes="States cashback rate = 5% at Amazon.com, Amazon Fresh, and Whole Foods Market AND provides an official-source URL supporting it.",
        add_ins="Confirm the 5% categories on Amazon or issuer official pages."
    )
    await _verify_simple_datapoint(
        evaluator, pay_save,
        node_base_id="amz_pay_fuel",
        program_key="amazon",
        field_key="fuel_savings",
        dp=ps.fuel_savings,
        claim_text=f"Amazon Prime fuel savings: {ps.fuel_savings.value}." if _has_value(ps.fuel_savings) else "",
        field_desc_for_nodes="States fuel savings = 10¢/gallon at bp, Amoco, and ampm (or as claimed) AND provides an official-source URL supporting it.",
        add_ins="Prefer Amazon official page describing the fuel program. Ensure cents-per-gallon and participating brands are supported."
    )

    # Additional Perks
    ap = info.additional_perks if info else AdditionalPerksInfo()
    await _verify_simple_datapoint(
        evaluator, perks,
        node_base_id="amz_perks_streaming",
        program_key="amazon",
        field_key="streaming",
        dp=ap.streaming,
        claim_text=f"Included streaming service for Amazon Prime: {ap.streaming.value}." if _has_value(ap.streaming) else "",
        field_desc_for_nodes="States included streaming service = Prime Video AND provides an official-source URL supporting it.",
        add_ins="Amazon official pages (including primevideo.com) count as official."
    )
    await _verify_simple_datapoint(
        evaluator, perks,
        node_base_id="amz_perks_pharmacy",
        program_key="amazon",
        field_key="pharmacy",
        dp=ap.pharmacy,
        claim_text=f"Amazon Prime pharmacy benefits: {ap.pharmacy.value}." if _has_value(ap.pharmacy) else "",
        field_desc_for_nodes="States pharmacy benefits include RxPass and/or free Two-Day shipping on prescriptions AND provides an official-source URL supporting it.",
        add_ins="Confirm RxPass price (e.g., $5/month if stated) and prescription shipping benefits on Amazon official pages."
    )
    await _verify_simple_datapoint(
        evaluator, perks,
        node_base_id="amz_perks_photos",
        program_key="amazon",
        field_key="photo_storage",
        dp=ap.photo_storage,
        claim_text=f"Amazon Prime photo storage benefit: {ap.photo_storage.value}." if _has_value(ap.photo_storage) else "",
        field_desc_for_nodes="States photo storage benefit = Amazon Photos (e.g., unlimited full-resolution) AND provides an official-source URL supporting it.",
        add_ins="Verify storage tier and 'unlimited full-resolution photos' if claimed, on Amazon official pages."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the 2026 retail membership comparison task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Overall programs are independent
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

    # Important: root cannot be critical if children have mixed criticality (framework constraint)
    # So we treat the root as non-critical here while category nodes enforce critical requirements.

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_memberships(),
        template_class=MembershipExtraction,
        extraction_name="membership_extraction"
    )

    # Build program subtrees
    await _verify_target_circle_360(
        evaluator,
        root,
        extracted.target_circle_360 if extracted else None
    )
    await _verify_walmart_plus(
        evaluator,
        root,
        extracted.walmart_plus if extracted else None
    )
    await _verify_amazon_prime(
        evaluator,
        root,
        extracted.amazon_prime if extracted else None
    )

    return evaluator.get_summary()