import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "airtag2_research"
TASK_DESCRIPTION = (
    "I am considering purchasing the Apple AirTag (2nd generation) and need to verify several technical "
    "specifications and details before making my decision. Please research and provide the following verified "
    "information about the AirTag 2nd generation: (1) The exact weight in grams, (2) The IP water and dust "
    "resistance rating, (3) The specific battery model type used, (4) Which generation of Apple's Ultra Wideband "
    "chip it contains, (5) The minimum iOS version required for compatibility, (6) The minimum Apple Watch Series "
    "model that supports Precision Finding functionality when running watchOS 26.2.1, (7) The retail price for "
    "purchasing a single unit in USD, (8) The official date when the AirTag 2nd generation became available for "
    "shipping/purchase, and (9) The percentage improvement in Precision Finding range compared to the 1st generation "
    "AirTag. For each piece of information, please provide the specific fact along with a reference URL from an "
    "official or authoritative source that verifies this information."
)

# Expectations explicitly stated by the rubric for some fields
EXPECTED = {
    "battery_model_must_include": "CR2032",
    "uwb_generation_must_indicate": "second-generation",  # accept variations like "2nd generation", “second generation”, or “U2”
    "min_ios_must_be": "iOS 26 or later",
    "min_watch_model_must_be": "Apple Watch Series 9 or later, or Apple Watch Ultra 2 or later",
    "availability_date_human": "January 28, 2026",
    "availability_date_iso": "2026-01-28",
    "range_improvement_percent": 50,
}

AUTHORITATIVE_BASE_DOMAINS = {
    # Official Apple
    "apple.com",
    # Well-known authoritative outlets
    "theverge.com",
    "engadget.com",
    "cnet.com",
    "techcrunch.com",
    "arstechnica.com",
    "macrumors.com",
    "9to5mac.com",
    "wired.com",
    "gsmarena.com",
    "tomsguide.com",
    "pcmag.com",
    "anandtech.com",
    "wsj.com",
    "bloomberg.com",
    "reuters.com",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AirTag2Specs(BaseModel):
    # (1) Weight
    weight_grams: Optional[str] = None
    weight_urls: List[str] = Field(default_factory=list)

    # (2) IP rating
    ip_rating: Optional[str] = None
    ip_urls: List[str] = Field(default_factory=list)

    # (3) Battery model
    battery_model: Optional[str] = None
    battery_urls: List[str] = Field(default_factory=list)

    # (4) UWB chip generation
    uwb_chip_generation: Optional[str] = None
    uwb_urls: List[str] = Field(default_factory=list)

    # (5) Minimum iOS version
    min_ios_version: Optional[str] = None
    min_ios_urls: List[str] = Field(default_factory=list)

    # (6) Minimum Apple Watch model (with watchOS 26.2.1 for Precision Finding)
    min_watch_model_precision_finding: Optional[str] = None
    min_watch_urls: List[str] = Field(default_factory=list)

    # (7) Single unit retail price in USD
    single_unit_price_usd: Optional[str] = None
    price_urls: List[str] = Field(default_factory=list)

    # (8) Official availability/shipping date
    availability_date: Optional[str] = None
    availability_urls: List[str] = Field(default_factory=list)

    # (9) Precision Finding range improvement percentage vs 1st gen
    range_improvement_percentage: Optional[str] = None
    range_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_airtag2_specs() -> str:
    return """
    You are extracting structured facts specifically for Apple AirTag (2nd generation) from the provided answer text.
    Extract the following fields as strings, exactly as written in the answer (do not normalize numbers other than
    copying what the answer wrote). If a field is not present in the answer, set it to null. For each fact, also
    extract the list of reference URLs that the answer explicitly cites for that fact. Only include URLs that appear
    in the answer.

    Required fields to extract:
    1) weight_grams: The exact weight in grams for AirTag (2nd generation). Include units if the answer includes them (e.g., "10.9 g" or "11 grams").
    2) weight_urls: All URLs the answer cites that support the weight.

    3) ip_rating: The water/dust resistance rating in IEC 60529 format, e.g., "IP67" or "IP68". Must be a single IP-formatted string if present.
    4) ip_urls: All URLs the answer cites that support the IP rating.

    5) battery_model: The specific coin-cell battery model/type used (e.g., "CR2032"). Do not just say "coin cell"; the answer must include the model if present.
    6) battery_urls: All URLs the answer cites that support the battery model.

    7) uwb_chip_generation: The generation of Apple's Ultra Wideband chip in AirTag (2nd generation) (e.g., "second-generation Ultra Wideband chip").
    8) uwb_urls: All URLs the answer cites that support the UWB chip generation.

    9) min_ios_version: The minimum iOS version required for compatibility with AirTag (2nd generation) (e.g., "iOS 26 or later").
    10) min_ios_urls: All URLs the answer cites that support the minimum iOS requirement.

    11) min_watch_model_precision_finding: The minimum Apple Watch model that supports Precision Finding on watchOS 26.2.1 for AirTag (2nd generation).
        For example: "Apple Watch Series 9 or later, or Apple Watch Ultra 2 or later".
    12) min_watch_urls: All URLs the answer cites that support the minimum Apple Watch model requirement.

    13) single_unit_price_usd: The retail price for purchasing a single AirTag (2nd generation) unit in USD (e.g., "$29" or "US$29").
    14) price_urls: All URLs the answer cites that support the price.

    15) availability_date: The official availability/shipping date (not merely announcement date) for AirTag (2nd generation), e.g., "January 28, 2026".
    16) availability_urls: All URLs the answer cites that support the availability/shipping date.

    17) range_improvement_percentage: The percentage improvement in Precision Finding range vs. 1st generation (e.g., "50%").
    18) range_urls: All URLs the answer cites that support the range improvement percentage.

    URL extraction rules:
    - Only include URLs explicitly present in the answer.
    - Extract full valid URLs (include http/https).
    - If none are cited for a fact, return an empty list for that fact's URLs.

    Output a single JSON object with all the fields above.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_grams_value(s: Optional[str]) -> bool:
    if not s:
        return False
    # Accept formats like "11 g", "10.9g", "11 grams"
    return bool(re.search(r"(?i)\b\d+(\.\d+)?\s*(g|grams?)\b", s.strip()))


def _is_ip_format(s: Optional[str]) -> bool:
    if not s:
        return False
    # Typical IP rating formats: IP67, IP68, IP6X, etc.
    return bool(re.match(r"(?i)^IP\d{2}[A-Z0-9]?$", s.strip()))


def _contains_cr2032(s: Optional[str]) -> bool:
    return bool(s and re.search(r"(?i)\bCR-?2032\b", s))


def _mentions_second_gen_uwb(s: Optional[str]) -> bool:
    if not s:
        return False
    s_low = s.lower()
    # Accept “second-generation”, “second generation”, “2nd generation”, or “U2”
    return ("second-generation" in s_low) or ("second generation" in s_low) or ("2nd generation" in s_low) or re.search(r"\bU2\b", s)


def _mentions_ios_26_or_later(s: Optional[str]) -> bool:
    if not s:
        return False
    return bool(re.search(r"(?i)\biOS\s*26(\.0)?\s*(or\s*later)?\b", s))


def _mentions_watch_s9_or_ultra2_or_later(s: Optional[str]) -> bool:
    if not s:
        return False
    s_low = s.lower()
    # Heuristics: must mention Series 9 (or S9) OR Ultra 2, ideally with "or later"
    cond_model = ("series 9" in s_low) or (re.search(r"\bS9\b", s_low) is not None) or ("ultra 2" in s_low)
    cond_later = ("or later" in s_low) or ("later" in s_low)
    return cond_model and cond_later


def _is_usd_price_format(s: Optional[str]) -> bool:
    if not s:
        return False
    # Accept $29, $29.00, US$29, US$ 29, 29 USD
    return bool(re.search(r"(?i)(US?\$\s?\d+(\.\d{2})?|\b\d+(\.\d{2})?\s?USD\b)", s.strip()))


def _parse_date_any(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    fmts = [
        "%B %d, %Y",       # January 28, 2026
        "%b %d, %Y",       # Jan 28, 2026
        "%Y-%m-%d",        # 2026-01-28
        "%m/%d/%Y",        # 01/28/2026
        "%d %B %Y",        # 28 January 2026
        "%d %b %Y",        # 28 Jan 2026
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except Exception:
            continue
    return None


def _matches_availability_date(s: Optional[str], expected_iso: str) -> bool:
    if not s:
        return False
    dt = _parse_date_any(s)
    if not dt:
        return False
    try:
        return dt.strftime("%Y-%m-%d") == expected_iso
    except Exception:
        return False


def _parse_percent_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    # Extract numeric like 50, from "50%" or "up to 50% range improvement" or "fifty percent" (we won't convert words)
    m = re.search(r"(\d{1,3})\s*%", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _is_authoritative_domain(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        # Strip port if any
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        # Check Apple (any subdomain of apple.com)
        if netloc.endswith("apple.com"):
            return True
        # Check other authoritative bases
        for base in AUTHORITATIVE_BASE_DOMAINS:
            if netloc == base or netloc.endswith("." + base):
                return True
        return False
    except Exception:
        return False


async def _verify_with_urls_or_fail(
    evaluator: Evaluator,
    node,
    claim: str,
    urls: List[str],
    add_ins: str,
) -> bool:
    # If no URLs, fail this leaf explicitly (since rubric requires citation)
    if not urls:
        node.score = 0.0
        node.status = "failed"
        return False
    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Build and verify tree                                                       #
# --------------------------------------------------------------------------- #
async def _build_and_verify(evaluator: Evaluator, root, ex: AirTag2Specs) -> None:
    # Root node (critical, parallel)
    # Already created as `root` in evaluate_answer

    # ============== Weight ==================
    weight_node = evaluator.add_parallel(
        id="Weight",
        desc="Provides the AirTag (2nd generation) weight and a supporting citation.",
        parent=root,
        critical=True
    )
    # Leaf: Value must be in grams
    evaluator.add_custom_node(
        result=_has_grams_value(ex.weight_grams),
        id="Weight_Value_Grams",
        desc="States the exact weight in grams (as specified in official Apple technical specifications).",
        parent=weight_node,
        critical=True
    )
    # Leaf: Citation URL supports the stated weight
    weight_cite_leaf = evaluator.add_leaf(
        id="Weight_Citation_URL",
        desc="Provides a reference URL (official Apple source or authoritative technology news source) that supports the stated weight.",
        parent=weight_node,
        critical=True
    )
    weight_claim = f"The AirTag (2nd generation) weighs {ex.weight_grams}."
    await _verify_with_urls_or_fail(
        evaluator,
        weight_cite_leaf,
        weight_claim,
        ex.weight_urls,
        add_ins="Only pass if the page is an official Apple page or a major authoritative technology news outlet, "
                "and it clearly refers to AirTag (2nd generation). Allow minor rounding differences (e.g., 10.9 g vs 11 g)."
    )

    # ============== IP Rating ==================
    ip_node = evaluator.add_parallel(
        id="IP_Rating",
        desc="Provides the IP water/dust resistance rating and a supporting citation.",
        parent=root,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_ip_format(ex.ip_rating),
        id="IP_Value_Format",
        desc="States the IP rating in IEC 60529 IPxx format.",
        parent=ip_node,
        critical=True
    )
    ip_cite_leaf = evaluator.add_leaf(
        id="IP_Citation_URL",
        desc="Provides a reference URL (official Apple source or authoritative technology news source) that supports the stated IP rating.",
        parent=ip_node,
        critical=True
    )
    ip_claim = f"The AirTag (2nd generation) has an IP rating of {ex.ip_rating} under IEC 60529."
    await _verify_with_urls_or_fail(
        evaluator,
        ip_cite_leaf,
        ip_claim,
        ex.ip_urls,
        add_ins="The page must clearly state the dust/water resistance rating for AirTag (2nd generation). "
                "Consider reasonable variations in phrasing like 'IP67 water and dust resistant'. "
                "Treat sources not from Apple or major authoritative outlets as insufficient."
    )

    # ============== Battery Model Type ==================
    batt_node = evaluator.add_parallel(
        id="Battery_Model_Type",
        desc="Provides the specific battery model/type used and a supporting citation.",
        parent=root,
        critical=True
    )
    evaluator.add_custom_node(
        result=_contains_cr2032(ex.battery_model),
        id="Battery_Model_Value_CR2032",
        desc="States the battery model designation/type as CR2032 coin cell (not merely 'coin cell').",
        parent=batt_node,
        critical=True
    )
    batt_cite_leaf = evaluator.add_leaf(
        id="Battery_Citation_URL",
        desc="Provides a reference URL (official Apple source or authoritative technology news source) that supports the stated battery model/type.",
        parent=batt_node,
        critical=True
    )
    batt_claim = "AirTag (2nd generation) uses a CR2032 coin cell battery."
    await _verify_with_urls_or_fail(
        evaluator,
        batt_cite_leaf,
        batt_claim,
        ex.battery_urls,
        add_ins="The page must explicitly mention CR2032 for AirTag (2nd generation). Mentions of 'coin cell' alone are insufficient."
    )

    # ============== UWB Chip Generation ==================
    uwb_node = evaluator.add_parallel(
        id="UWB_Chip_Generation",
        desc="Provides which generation of Apple's Ultra Wideband chip it contains and a supporting citation.",
        parent=root,
        critical=True
    )
    evaluator.add_custom_node(
        result=_mentions_second_gen_uwb(ex.uwb_chip_generation),
        id="UWB_Generation_Value_Second_Gen",
        desc="Identifies the Ultra Wideband chip as Apple's second-generation UWB chip.",
        parent=uwb_node,
        critical=True
    )
    uwb_cite_leaf = evaluator.add_leaf(
        id="UWB_Citation_URL",
        desc="Provides a reference URL (official Apple source or authoritative technology news source) that supports the stated UWB chip generation.",
        parent=uwb_node,
        critical=True
    )
    uwb_claim = "AirTag (2nd generation) contains Apple's second-generation Ultra Wideband chip."
    await _verify_with_urls_or_fail(
        evaluator,
        uwb_cite_leaf,
        uwb_claim,
        ex.uwb_urls,
        add_ins="If the page mentions U1 or 'first-generation' only, mark as unsupported. "
                "Accept variants like 'second‑generation Ultra Wideband chip' or 'U2 chip'."
    )

    # ============== Minimum iOS Compatibility ==================
    ios_node = evaluator.add_parallel(
        id="Minimum_iOS_Compatibility",
        desc="Provides the minimum iOS version required and a supporting citation.",
        parent=root,
        critical=True
    )
    evaluator.add_custom_node(
        result=_mentions_ios_26_or_later(ex.min_ios_version),
        id="Min_iOS_Value_iOS26_or_later",
        desc="States the minimum iOS version required as iOS 26.0 or later.",
        parent=ios_node,
        critical=True
    )
    ios_cite_leaf = evaluator.add_leaf(
        id="Min_iOS_Citation_URL",
        desc="Provides a reference URL (official Apple source or authoritative technology news source) that supports the stated minimum iOS version requirement.",
        parent=ios_node,
        critical=True
    )
    ios_claim = "The minimum iOS version required for AirTag (2nd generation) compatibility is iOS 26 or later."
    await _verify_with_urls_or_fail(
        evaluator,
        ios_cite_leaf,
        ios_claim,
        ex.min_ios_urls,
        add_ins="The page must clearly indicate 'iOS 26' (or 'iOS 26.0') as the minimum requirement for AirTag (2nd generation). "
                "Mentions of older iOS versions (e.g., 17/18) are insufficient."
    )

    # ============== Apple Watch Precision Finding Minimum Model ==================
    watch_node = evaluator.add_parallel(
        id="Apple_Watch_Precision_Finding_Minimum_Model",
        desc="Provides the minimum Apple Watch model supporting Precision Finding when running watchOS 26.2.1 and a supporting citation.",
        parent=root,
        critical=True
    )
    evaluator.add_custom_node(
        result=_mentions_watch_s9_or_ultra2_or_later(ex.min_watch_model_precision_finding),
        id="Min_Watch_Model_Value_S9_or_Ultra2_or_later",
        desc="States that Precision Finding on Apple Watch with watchOS 26.2.1 requires Apple Watch Series 9 or later, or Apple Watch Ultra 2 or later.",
        parent=watch_node,
        critical=True
    )
    watch_cite_leaf = evaluator.add_leaf(
        id="Min_Watch_Model_Citation_URL",
        desc="Provides a reference URL (official Apple source or authoritative technology news source) that supports the stated minimum Apple Watch model requirement.",
        parent=watch_node,
        critical=True
    )
    watch_claim = (
        "Precision Finding on Apple Watch with watchOS 26.2.1 requires Apple Watch Series 9 or later, "
        "or Apple Watch Ultra 2 or later."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        watch_cite_leaf,
        watch_claim,
        ex.min_watch_urls,
        add_ins="The page must explicitly tie Precision Finding for AirTag (2nd generation) on watchOS 26.2.1 to "
                "Apple Watch Series 9 (or later) or Ultra 2 (or later)."
    )

    # ============== Single Unit Retail Price (USD) ==================
    price_node = evaluator.add_parallel(
        id="Single_Unit_Retail_Price_USD",
        desc="Provides the retail price for a single unit in USD and a supporting citation.",
        parent=root,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_usd_price_format(ex.single_unit_price_usd),
        id="Price_Value_USD",
        desc="States the retail price for purchasing a single unit in USD.",
        parent=price_node,
        critical=True
    )
    price_cite_leaf = evaluator.add_leaf(
        id="Price_Citation_URL",
        desc="Provides a reference URL (official Apple source or authoritative technology news source) that supports the stated single-unit USD price.",
        parent=price_node,
        critical=True
    )
    price_claim = f"The retail price for a single AirTag (2nd generation) unit is {ex.single_unit_price_usd} in USD."
    await _verify_with_urls_or_fail(
        evaluator,
        price_cite_leaf,
        price_claim,
        ex.price_urls,
        add_ins="Confirm the U.S. retail price (USD) for a single AirTag (2nd generation) unit on an official Apple page "
                "or a major authoritative outlet quoting Apple pricing."
    )

    # ============== Official Availability / Shipping Date ==================
    avail_node = evaluator.add_parallel(
        id="Official_Availability_Date",
        desc="Provides the official shipping/purchase availability date and a supporting citation.",
        parent=root,
        critical=True
    )
    evaluator.add_custom_node(
        result=_matches_availability_date(ex.availability_date, EXPECTED["availability_date_iso"]),
        id="Availability_Date_Value_Jan_28_2026",
        desc="States the official availability/shipping date as January 28, 2026 (and treats it as availability/shipping, not merely the announcement date).",
        parent=avail_node,
        critical=True
    )
    avail_cite_leaf = evaluator.add_leaf(
        id="Availability_Date_Citation_URL",
        desc="Provides a reference URL (official Apple source or authoritative technology news source) that supports the stated availability/shipping date.",
        parent=avail_node,
        critical=True
    )
    avail_claim = "AirTag (2nd generation) became available for shipping/purchase on January 28, 2026."
    await _verify_with_urls_or_fail(
        evaluator,
        avail_cite_leaf,
        avail_claim,
        ex.availability_urls,
        add_ins="Only pass if the page explicitly states the availability or shipping date (not merely the announcement) "
                "for AirTag (2nd generation) as January 28, 2026."
    )

    # ============== Precision Finding Range Improvement ==================
    range_node = evaluator.add_parallel(
        id="Precision_Finding_Range_Improvement",
        desc="Provides the percentage improvement in Precision Finding range vs. 1st generation and a supporting citation.",
        parent=root,
        critical=True
    )
    evaluator.add_custom_node(
        result=(_parse_percent_int(ex.range_improvement_percentage) == EXPECTED["range_improvement_percent"]),
        id="Range_Improvement_Percentage_Value_50",
        desc="States the Precision Finding range improvement as 50% compared to the 1st generation AirTag.",
        parent=range_node,
        critical=True
    )
    range_cite_leaf = evaluator.add_leaf(
        id="Range_Improvement_Citation_URL",
        desc="Provides a reference URL (official Apple source or authoritative technology news source) that supports the stated percentage improvement.",
        parent=range_node,
        critical=True
    )
    range_claim = (
        "Precision Finding range for AirTag (2nd generation) is improved by 50% compared to the first generation AirTag."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        range_cite_leaf,
        range_claim,
        ex.range_urls,
        add_ins="The page must clearly quantify a 50% improvement in Precision Finding range compared to 1st gen. "
                "Accept phrasing like 'up to 50% more range'."
    )

    # --------- Additional: record simple authoritative URL presence stats (non-evaluative) ----------
    stats = {
        "authoritative_url_presence": {
            "weight": any(_is_authoritative_domain(u) for u in ex.weight_urls),
            "ip": any(_is_authoritative_domain(u) for u in ex.ip_urls),
            "battery": any(_is_authoritative_domain(u) for u in ex.battery_urls),
            "uwb": any(_is_authoritative_domain(u) for u in ex.uwb_urls),
            "min_ios": any(_is_authoritative_domain(u) for u in ex.min_ios_urls),
            "min_watch": any(_is_authoritative_domain(u) for u in ex.min_watch_urls),
            "price": any(_is_authoritative_domain(u) for u in ex.price_urls),
            "availability": any(_is_authoritative_domain(u) for u in ex.availability_urls),
            "range": any(_is_authoritative_domain(u) for u in ex.range_urls),
        }
    }
    evaluator.add_custom_info(stats, info_type="diagnostics", info_name="authoritativeness_check")


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
    """
    Evaluate an answer for the AirTag (2nd generation) research task using the Mind2Web2 framework.
    Returns a structured summary dictionary containing the verification tree and final score.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent criteria; all are critical
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

    # Override root node to be critical (the rubric marks the root critical)
    root.critical = True
    root.desc = (
        "Provides all required AirTag (2nd generation) specifications/details requested, and for each requested fact "
        "includes a URL citation from an official Apple source or an authoritative technology news source that supports that specific fact."
    )

    # Extract structured information from the answer
    extracted: AirTag2Specs = await evaluator.extract(
        prompt=prompt_extract_airtag2_specs(),
        template_class=AirTag2Specs,
        extraction_name="airtag_2nd_gen_extraction",
    )

    # Ground truth/expectations (for transparency; not directly enforcing truth here other than custom checks above)
    evaluator.add_ground_truth(
        {
            "expected_constraints": EXPECTED,
            "notes": "Some criteria explicitly require certain values (e.g., iOS 26+, second‑generation UWB, CR2032, Jan 28 2026, 50%). "
                     "Leaf nodes check answer text formatting/content where applicable, while citation leaves verify the claim via the provided sources."
        },
        gt_type="expectations"
    )

    # Build and verify all rubric branches
    await _build_and_verify(evaluator, root, extracted)

    # Return evaluator summary
    return evaluator.get_summary()