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
TASK_ID = "smartphone_2026_tri_carrier_reliability"
TASK_DESCRIPTION = """
In early 2026, a corporate mobile technology consultant is advising a multinational client whose executives travel frequently across Texas, Georgia, New York, and California for business operations. Following the January 14, 2026 Verizon network outage that affected these states for approximately 10 hours and put millions of devices into SOS-only mode, the client demands maximum network reliability and seamless tri-carrier compatibility. The consultant must identify a smartphone model commercially available in the United States in 2026 that satisfies ALL of the following technical requirements: **LTE Band Support Requirements:** Must support LTE bands 2 (1900 MHz), 4 (1700/2100 MHz), 5 (850 MHz), 12 (700 MHz), 13 (700 MHz), 66 (1700/2100 MHz AWS-3), and 71 (600 MHz). **5G NR Band Support Requirements:** Must support 5G NR bands n2, n5, n41 (2.5 GHz), n66, n71 (600 MHz), and n77 (3.7 GHz C-band). **Carrier Feature Certification:** Must be VoLTE-certified and support WiFi calling functionality on all three major carriers: Verizon, AT&T, and T-Mobile. **Device Features:** Must support dual SIM functionality (either physical SIM + eSIM or dual eSIM configuration). **Device Status:** Must be sold carrier-unlocked or be eligible for unlocking under current FCC and carrier policies, must have clean IMEI status (not blacklisted), and must be FCC certified for use in the United States. **Geographic Coverage:** Must provide reliable cellular network coverage in all four target states: Texas, Georgia, New York, and California. Provide the specific device model name, manufacturer, and supporting documentation URLs that verify compliance with each technical requirement category.
"""

REQUIRED_LTE_BANDS = ["2", "4", "5", "12", "13", "66", "71"]
REQUIRED_NR_BANDS = ["n2", "n5", "n41", "n66", "n71", "n77"]
REQUIRED_STATES = ["Texas", "Georgia", "New York", "California"]
REQUIRED_CARRIERS = ["Verizon", "AT&T", "T-Mobile"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CarrierSupportURLs(BaseModel):
    volte: List[str] = Field(default_factory=list)
    wifi_calling: List[str] = Field(default_factory=list)


class TriCarrierSupport(BaseModel):
    verizon: CarrierSupportURLs = Field(default_factory=CarrierSupportURLs)
    att: CarrierSupportURLs = Field(default_factory=CarrierSupportURLs)
    tmobile: CarrierSupportURLs = Field(default_factory=CarrierSupportURLs)


class SmartphoneExtraction(BaseModel):
    manufacturer: Optional[str] = None
    model_name: Optional[str] = None

    availability_statement: Optional[str] = None
    availability_urls: List[str] = Field(default_factory=list)

    # Band support documentation
    lte_band_urls: List[str] = Field(default_factory=list)
    nr_band_urls: List[str] = Field(default_factory=list)

    # Carrier feature documentation per carrier and feature
    tri_carrier_urls: TriCarrierSupport = Field(default_factory=TriCarrierSupport)

    # Device features and status documentation
    dual_sim_urls: List[str] = Field(default_factory=list)
    unlock_or_unlock_eligible_urls: List[str] = Field(default_factory=list)
    fcc_cert_urls: List[str] = Field(default_factory=list)
    imei_check_urls: List[str] = Field(default_factory=list)

    # Coverage documentation and discussion
    coverage_urls: List[str] = Field(default_factory=list)
    coverage_discussion: Optional[str] = None  # Extracted text discussing TX, GA, NY, CA coverage


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_smartphone() -> str:
    return """
Extract from the answer the exact device identification and all cited documentation URLs grouped by requirement category. Follow strictly:

1) Device Identification
- manufacturer: String with the device manufacturer (e.g., "Samsung", "Apple", "Google").
- model_name: String with the specific device model name (e.g., "Galaxy S24 Ultra", "iPhone 15 Pro", "Pixel 9 Pro").

2) U.S. Availability in 2026
- availability_statement: The exact sentence(s) from the answer that claim U.S. commercial availability in 2026; null if not stated.
- availability_urls: All URLs provided that substantiate U.S. availability in 2026 (official product pages, US retailers, etc.). Return empty list if none.

3) Band Support Documentation (device variant sold/usable in the United States)
- lte_band_urls: All URLs that document LTE band support (target bands include 2/4/5/12/13/66/71). Return only URLs explicitly present in the answer.
- nr_band_urls: All URLs that document 5G NR band support (target bands include n2/n5/n41/n66/n71/n77). Return only URLs explicitly present.

4) Tri‑Carrier Feature Documentation (per carrier)
Return URLs under tri_carrier_urls for each carrier that substantiate BOTH VoLTE and Wi‑Fi Calling compatibility. Only include URLs explicitly present in the answer. If none for a bucket, leave it empty.
- verizon.volte: URLs for Verizon VoLTE certification/compatibility for this model
- verizon.wifi_calling: URLs for Verizon Wi‑Fi Calling
- att.volte: URLs for AT&T VoLTE/HD Voice certification/compatibility
- att.wifi_calling: URLs for AT&T Wi‑Fi Calling
- tmobile.volte: URLs for T‑Mobile VoLTE
- tmobile.wifi_calling: URLs for T‑Mobile Wi‑Fi Calling

5) Device Features & Status Documentation
- dual_sim_urls: URLs proving dual‑SIM (physical+eSIM OR dual eSIM) support
- unlock_or_unlock_eligible_urls: URLs proving the device is sold unlocked OR is eligible for unlocking under current US policies
- fcc_cert_urls: URLs proving FCC certification for US operation (e.g., FCC ID database page for the model/variant)
- imei_check_urls: URLs to official resources for checking IMEI status/blacklist (carrier or industry association)

6) Coverage Documentation & Discussion
- coverage_urls: URLs to carrier coverage maps/resources used to support coverage across Texas, Georgia, New York, and California
- coverage_discussion: The exact sentence(s) from the answer that discuss state‑level coverage for the four states. If missing, return null.

IMPORTANT:
- Extract ONLY URLs explicitly present in the answer (plain URLs or markdown links). Do not invent or infer URLs.
- If a requested URL category is not present, return an empty list for that category.
- If a requested text field is not present, return null for that field.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0


def _flatten_tri_carrier(tri: TriCarrierSupport) -> List[str]:
    all_urls = []
    all_urls.extend(tri.verizon.volte)
    all_urls.extend(tri.verizon.wifi_calling)
    all_urls.extend(tri.att.volte)
    all_urls.extend(tri.att.wifi_calling)
    all_urls.extend(tri.tmobile.volte)
    all_urls.extend(tri.tmobile.wifi_calling)
    # De-duplicate while preserving order
    seen = set()
    dedup = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_device_identification_and_availability(
    evaluator: Evaluator,
    parent,
    data: SmartphoneExtraction,
):
    node = evaluator.add_parallel(
        id="Device_Identification_And_Availability",
        desc="Answer identifies the device and confirms US commercial availability in 2026.",
        parent=parent,
        critical=True,
    )

    # Model name and manufacturer provided (existence check)
    evaluator.add_custom_node(
        result=bool(data.manufacturer and data.manufacturer.strip() and data.model_name and data.model_name.strip()),
        id="Model_Name_And_Manufacturer_Provided",
        desc="Provides the specific device model name and manufacturer.",
        parent=node,
        critical=True,
    )

    # Commercial availability in US in 2026 (try to ground with URLs if provided)
    avail_leaf = evaluator.add_leaf(
        id="Commercially_Available_US_2026",
        desc="States the device is commercially available for purchase in the United States in 2026.",
        parent=node,
        critical=True,
    )
    availability_claim = (
        f"The device {data.manufacturer or ''} {data.model_name or ''} "
        f"is commercially available for purchase in the United States in 2026."
    ).strip()

    await evaluator.verify(
        claim=availability_claim,
        node=avail_leaf,
        sources=data.availability_urls if _nonempty(data.availability_urls) else None,
        additional_instruction=(
            "If URLs are provided, verify they indicate US commercial availability (e.g., US product page or US retailer "
            "that sold/sells the device in 2026). If no URLs, verify the answer text actually asserts US availability in 2026."
        ),
    )


async def verify_radio_access_bands(
    evaluator: Evaluator,
    parent,
    data: SmartphoneExtraction,
):
    node = evaluator.add_parallel(
        id="Radio_Access_Band_Compatibility",
        desc="Device supports all required LTE and 5G NR bands listed in the question.",
        parent=parent,
        critical=True,
    )

    # LTE bands
    lte_leaf = evaluator.add_leaf(
        id="LTE_Bands_All_Required_Supported",
        desc="States the device supports LTE bands 2, 4, 5, 12, 13, 66, and 71.",
        parent=node,
        critical=True,
    )
    lte_claim = (
        f"This device supports LTE bands {', '.join(REQUIRED_LTE_BANDS)} "
        f"(including AWS-3 for band 66 and 600 MHz for band 71) for the US variant."
    )
    await evaluator.verify(
        claim=lte_claim,
        node=lte_leaf,
        sources=data.lte_band_urls if _nonempty(data.lte_band_urls) else None,
        additional_instruction=(
            "Accept if the documentation for this exact model/US variant explicitly lists all of these LTE bands: "
            "2, 4, 5, 12, 13, 66, and 71. Allow equivalent notations such as 'B2', 'LTE FDD band 2', 'PCS 1900 (B2)', "
            "'AWS-3 (B66)', and '600 MHz (B71)'. If multiple SKUs exist, ensure the cited one covers the US bands."
        ),
    )

    # 5G NR bands
    nr_leaf = evaluator.add_leaf(
        id="NR5G_Bands_All_Required_Supported",
        desc="States the device supports 5G NR bands n2, n5, n41, n66, n71, and n77.",
        parent=node,
        critical=True,
    )
    nr_claim = (
        f"This device supports 5G NR bands {', '.join(REQUIRED_NR_BANDS)} "
        f"(including 2.5 GHz for n41 and 3.7 GHz C-band for n77) for the US variant."
    )
    await evaluator.verify(
        claim=nr_claim,
        node=nr_leaf,
        sources=data.nr_band_urls if _nonempty(data.nr_band_urls) else None,
        additional_instruction=(
            "Accept if the documentation for this exact model/US variant explicitly lists ALL required NR bands: "
            "n2, n5, n41, n66, n71, n77. Allow common synonyms/notes (e.g., 'C-band' for n77)."
        ),
    )


async def verify_tri_carrier_features(
    evaluator: Evaluator,
    parent,
    data: SmartphoneExtraction,
):
    node = evaluator.add_parallel(
        id="Tri_Carrier_Features",
        desc="Device supports required carrier features on Verizon, AT&T, and T-Mobile.",
        parent=parent,
        critical=True,
    )

    # VoLTE - split per carrier to enforce all three carriers
    volte_node = evaluator.add_parallel(
        id="VoLTE_Certified_All_Three_Carriers",
        desc="States the device is VoLTE-compatible/certified on Verizon, AT&T, and T-Mobile.",
        parent=node,
        critical=True,
    )

    # Verizon VoLTE
    vzw_volte = evaluator.add_leaf(
        id="VoLTE_Verizon",
        desc="Device is VoLTE-compatible/certified on Verizon.",
        parent=volte_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This device is VoLTE-compatible/certified on Verizon (a.k.a. HD Voice/Advanced Calling over LTE).",
        node=vzw_volte,
        sources=data.tri_carrier_urls.verizon.volte if _nonempty(data.tri_carrier_urls.verizon.volte) else None,
        additional_instruction=(
            "Accept references to Verizon's official whitelist/compatibility pages, manufacturer or carrier pages explicitly "
            "stating Verizon VoLTE/HD Voice support for this model. Synonyms allowed: 'VoLTE', 'HD Voice', 'Advanced Calling'."
        ),
    )

    # AT&T VoLTE
    att_volte = evaluator.add_leaf(
        id="VoLTE_ATT",
        desc="Device is VoLTE-compatible/certified on AT&T.",
        parent=volte_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This device is VoLTE/HD Voice compatible/certified on AT&T.",
        node=att_volte,
        sources=data.tri_carrier_urls.att.volte if _nonempty(data.tri_carrier_urls.att.volte) else None,
        additional_instruction=(
            "Accept AT&T whitelist/compatibility or manufacturer pages. Synonyms: 'VoLTE', 'HD Voice', 'Enhanced LTE'. "
            "Ensure the page ties to this model."
        ),
    )

    # T-Mobile VoLTE
    tmo_volte = evaluator.add_leaf(
        id="VoLTE_TMobile",
        desc="Device is VoLTE-compatible/certified on T-Mobile.",
        parent=volte_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This device is VoLTE-compatible/certified on T-Mobile.",
        node=tmo_volte,
        sources=data.tri_carrier_urls.tmobile.volte if _nonempty(data.tri_carrier_urls.tmobile.volte) else None,
        additional_instruction=(
            "Accept T-Mobile official support/compatibility or manufacturer pages that clearly state VoLTE support "
            "for this model."
        ),
    )

    # Wi-Fi Calling - split per carrier to enforce all three carriers
    wifi_node = evaluator.add_parallel(
        id="WiFi_Calling_Supported_All_Three_Carriers",
        desc="States the device supports WiFi calling on Verizon, AT&T, and T-Mobile.",
        parent=node,
        critical=True,
    )

    # Verizon Wi-Fi Calling
    vzw_wifi = evaluator.add_leaf(
        id="WiFiCalling_Verizon",
        desc="Device supports Wi-Fi Calling on Verizon.",
        parent=wifi_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This device supports Wi‑Fi Calling on Verizon (a.k.a. VoWiFi).",
        node=vzw_wifi,
        sources=data.tri_carrier_urls.verizon.wifi_calling if _nonempty(data.tri_carrier_urls.verizon.wifi_calling) else None,
        additional_instruction=(
            "Accept Verizon official support pages or manufacturer documentation explicitly listing Wi‑Fi Calling/VoWiFi for this model."
        ),
    )

    # AT&T Wi-Fi Calling
    att_wifi = evaluator.add_leaf(
        id="WiFiCalling_ATT",
        desc="Device supports Wi‑Fi Calling on AT&T.",
        parent=wifi_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This device supports Wi‑Fi Calling on AT&T (VoWiFi).",
        node=att_wifi,
        sources=data.tri_carrier_urls.att.wifi_calling if _nonempty(data.tri_carrier_urls.att.wifi_calling) else None,
        additional_instruction=(
            "Accept AT&T official support or manufacturer pages stating Wi‑Fi Calling/VoWiFi support for this model."
        ),
    )

    # T-Mobile Wi-Fi Calling
    tmo_wifi = evaluator.add_leaf(
        id="WiFiCalling_TMobile",
        desc="Device supports Wi‑Fi Calling on T‑Mobile.",
        parent=wifi_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This device supports Wi‑Fi Calling on T‑Mobile (VoWiFi).",
        node=tmo_wifi,
        sources=data.tri_carrier_urls.tmobile.wifi_calling if _nonempty(data.tri_carrier_urls.tmobile.wifi_calling) else None,
        additional_instruction=(
            "Accept T‑Mobile official support or manufacturer pages stating Wi‑Fi Calling/VoWiFi support for this model."
        ),
    )


async def verify_device_features_and_status(
    evaluator: Evaluator,
    parent,
    data: SmartphoneExtraction,
):
    node = evaluator.add_parallel(
        id="Device_Features_And_Status",
        desc="Device supports dual SIM and satisfies unlock/IMEI/FCC status requirements.",
        parent=parent,
        critical=True,
    )

    # Dual SIM
    dual_sim_leaf = evaluator.add_leaf(
        id="Dual_SIM_Supported",
        desc="States the device supports dual SIM (physical SIM + eSIM or dual eSIM).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This device supports dual SIM functionality, either physical SIM + eSIM or dual eSIM (two lines).",
        node=dual_sim_leaf,
        sources=data.dual_sim_urls if _nonempty(data.dual_sim_urls) else None,
        additional_instruction="Accept manufacturer spec or carrier support pages clearly indicating dual SIM or dual eSIM capability.",
    )

    # Unlocked / unlock-eligible
    unlock_leaf = evaluator.add_leaf(
        id="Unlocked_Or_Unlock_Eligible",
        desc="States the device is sold carrier-unlocked OR is eligible for unlocking under current FCC/carrier policies.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This device is either sold carrier‑unlocked in the US or is eligible for unlocking under US FCC/carrier policies.",
        node=unlock_leaf,
        sources=data.unlock_or_unlock_eligible_urls if _nonempty(data.unlock_or_unlock_eligible_urls) else None,
        additional_instruction="Accept official product pages or carrier policy pages confirming unlocked availability or unlock eligibility for this model.",
    )

    # Clean IMEI requirement explicitly addressed in answer text
    clean_imei_leaf = evaluator.add_leaf(
        id="Clean_IMEI_Not_Blacklisted",
        desc="Addresses the clean-IMEI requirement (not blacklisted/lost/stolen) in a way that can be checked from the answer text (e.g., explicitly states the requirement applies to the unit obtained).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer explicitly requires that the specific purchased unit must have a clean IMEI "
            "(not blacklisted/lost/stolen) and mentions verifying IMEI status."
        ),
        node=clean_imei_leaf,
        sources=None,
        additional_instruction=(
            "Judge only from the answer text: look for explicit phrases such as 'clean IMEI', 'not blacklisted', "
            "'IMEI check', or instructions to verify IMEI status for the actual unit."
        ),
    )

    # FCC certification
    fcc_leaf = evaluator.add_leaf(
        id="FCC_Certified_For_US_Use",
        desc="States the device is FCC certified for operation in the United States.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This device is FCC certified for operation in the United States (FCC ID available/approved).",
        node=fcc_leaf,
        sources=data.fcc_cert_urls if _nonempty(data.fcc_cert_urls) else None,
        additional_instruction=(
            "Prefer the FCC ID database page for the exact model/variant or manufacturer documentation explicitly mentioning FCC certification."
        ),
    )


async def verify_geographic_coverage(
    evaluator: Evaluator,
    parent,
    data: SmartphoneExtraction,
):
    node = evaluator.add_parallel(
        id="Geographic_Coverage",
        desc="Answer addresses that the device can be used with major-carrier service in each of the four target states.",
        parent=parent,
        critical=True,
    )

    # Check the answer text explicitly mentions all four states in the coverage discussion
    coverage_text_leaf = evaluator.add_leaf(
        id="Coverage_In_All_Four_States_Addressed",
        desc="Explicitly addresses service/coverage in Texas, Georgia, New York, and California (all four states are individually mentioned in the answer’s coverage discussion).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer explicitly discusses coverage/service for each of these states individually: Texas, Georgia, New York, and California."
        ),
        node=coverage_text_leaf,
        sources=None,
        additional_instruction=(
            "Judge only from the answer text. Accept reasonable abbreviations like TX, GA, NY, CA if clearly referring to the states."
        ),
    )


async def verify_supporting_documentation_urls(
    evaluator: Evaluator,
    parent,
    data: SmartphoneExtraction,
):
    node = evaluator.add_parallel(
        id="Supporting_Documentation_URLs",
        desc="Provides supporting documentation URL(s) that substantiate each technical requirement category in the question.",
        parent=parent,
        critical=True,
    )

    # LTE band support URLs: existence + support
    lte_urls_exist = evaluator.add_custom_node(
        result=_nonempty(data.lte_band_urls),
        id="URLs_For_LTE_Band_Support_Provided",
        desc="Provides at least one URL documenting LTE band support.",
        parent=node,
        critical=True,
    )
    lte_urls_support = evaluator.add_leaf(
        id="URLs_For_LTE_Band_Support",
        desc="Provides URL(s) documenting LTE band support (including bands 2/4/5/12/13/66/71).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="These documentation URLs show that the device supports LTE bands 2, 4, 5, 12, 13, 66, and 71.",
        node=lte_urls_support,
        sources=data.lte_band_urls if _nonempty(data.lte_band_urls) else None,
        additional_instruction="All listed LTE bands must be covered for the US variant of the device.",
    )

    # 5G NR band support URLs: existence + support
    nr_urls_exist = evaluator.add_custom_node(
        result=_nonempty(data.nr_band_urls),
        id="URLs_For_5G_NR_Band_Support_Provided",
        desc="Provides at least one URL documenting 5G NR band support.",
        parent=node,
        critical=True,
    )
    nr_urls_support = evaluator.add_leaf(
        id="URLs_For_5G_NR_Band_Support",
        desc="Provides URL(s) documenting 5G NR band support (including n2/n5/n41/n66/n71/n77).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="These documentation URLs show that the device supports NR bands n2, n5, n41, n66, n71, and n77.",
        node=nr_urls_support,
        sources=data.nr_band_urls if _nonempty(data.nr_band_urls) else None,
        additional_instruction="All listed NR bands must be covered for the US variant of the device.",
    )

    # Carrier features URLs: split per carrier+feature to avoid false positives
    carrier_urls_node = evaluator.add_parallel(
        id="URLs_For_Carrier_Features",
        desc="Provides URL(s) supporting tri-carrier VoLTE and WiFi calling compatibility (Verizon, AT&T, T-Mobile).",
        parent=node,
        critical=True,
    )

    # Verizon URLs (VoLTE + Wi‑Fi Calling)
    vzw_volte_urls_leaf = evaluator.add_leaf(
        id="URLs_Verizon_VoLTE",
        desc="Provides URL(s) supporting Verizon VoLTE for the device.",
        parent=carrier_urls_node,
        critical=True,
    )
    await evaluator.verify(
        claim="These URLs substantiate Verizon VoLTE/HD Voice compatibility for this device.",
        node=vzw_volte_urls_leaf,
        sources=data.tri_carrier_urls.verizon.volte if _nonempty(data.tri_carrier_urls.verizon.volte) else None,
        additional_instruction="Prefer Verizon official whitelist/support or manufacturer pages listing Verizon VoLTE.",
    )
    vzw_wifi_urls_leaf = evaluator.add_leaf(
        id="URLs_Verizon_WiFiCalling",
        desc="Provides URL(s) supporting Verizon Wi‑Fi Calling for the device.",
        parent=carrier_urls_node,
        critical=True,
    )
    await evaluator.verify(
        claim="These URLs substantiate Verizon Wi‑Fi Calling (VoWiFi) compatibility for this device.",
        node=vzw_wifi_urls_leaf,
        sources=data.tri_carrier_urls.verizon.wifi_calling if _nonempty(data.tri_carrier_urls.verizon.wifi_calling) else None,
        additional_instruction="Prefer Verizon official support or manufacturer pages listing Wi‑Fi Calling support.",
    )

    # AT&T URLs
    att_volte_urls_leaf = evaluator.add_leaf(
        id="URLs_ATT_VoLTE",
        desc="Provides URL(s) supporting AT&T VoLTE/HD Voice for the device.",
        parent=carrier_urls_node,
        critical=True,
    )
    await evaluator.verify(
        claim="These URLs substantiate AT&T VoLTE/HD Voice compatibility for this device.",
        node=att_volte_urls_leaf,
        sources=data.tri_carrier_urls.att.volte if _nonempty(data.tri_carrier_urls.att.volte) else None,
        additional_instruction="Prefer AT&T whitelist/support or manufacturer pages listing AT&T VoLTE/HD Voice.",
    )
    att_wifi_urls_leaf = evaluator.add_leaf(
        id="URLs_ATT_WiFiCalling",
        desc="Provides URL(s) supporting AT&T Wi‑Fi Calling for the device.",
        parent=carrier_urls_node,
        critical=True,
    )
    await evaluator.verify(
        claim="These URLs substantiate AT&T Wi‑Fi Calling (VoWiFi) compatibility for this device.",
        node=att_wifi_urls_leaf,
        sources=data.tri_carrier_urls.att.wifi_calling if _nonempty(data.tri_carrier_urls.att.wifi_calling) else None,
        additional_instruction="Prefer AT&T official support or manufacturer pages listing Wi‑Fi Calling support.",
    )

    # T‑Mobile URLs
    tmo_volte_urls_leaf = evaluator.add_leaf(
        id="URLs_TMobile_VoLTE",
        desc="Provides URL(s) supporting T‑Mobile VoLTE for the device.",
        parent=carrier_urls_node,
        critical=True,
    )
    await evaluator.verify(
        claim="These URLs substantiate T‑Mobile VoLTE compatibility for this device.",
        node=tmo_volte_urls_leaf,
        sources=data.tri_carrier_urls.tmobile.volte if _nonempty(data.tri_carrier_urls.tmobile.volte) else None,
        additional_instruction="Prefer T‑Mobile official support or manufacturer pages listing T‑Mobile VoLTE.",
    )
    tmo_wifi_urls_leaf = evaluator.add_leaf(
        id="URLs_TMobile_WiFiCalling",
        desc="Provides URL(s) supporting T‑Mobile Wi‑Fi Calling for the device.",
        parent=carrier_urls_node,
        critical=True,
    )
    await evaluator.verify(
        claim="These URLs substantiate T‑Mobile Wi‑Fi Calling (VoWiFi) compatibility for this device.",
        node=tmo_wifi_urls_leaf,
        sources=data.tri_carrier_urls.tmobile.wifi_calling if _nonempty(data.tri_carrier_urls.tmobile.wifi_calling) else None,
        additional_instruction="Prefer T‑Mobile official support or manufacturer pages listing Wi‑Fi Calling support.",
    )

    # Dual-SIM URLs
    dualsim_urls_exist = evaluator.add_custom_node(
        result=_nonempty(data.dual_sim_urls),
        id="URLs_For_Dual_SIM_Provided",
        desc="Provides at least one URL supporting dual‑SIM capability.",
        parent=node,
        critical=True,
    )
    dualsim_urls_support = evaluator.add_leaf(
        id="URLs_For_Dual_SIM",
        desc="Provides URL(s) supporting dual-SIM capability (physical+eSIM or dual eSIM).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="These URLs substantiate dual‑SIM capability for the device (either physical+eSIM or dual eSIM).",
        node=dualsim_urls_support,
        sources=data.dual_sim_urls if _nonempty(data.dual_sim_urls) else None,
        additional_instruction="Prefer manufacturer specs or official carrier documentation.",
    )

    # Device status URLs: split into unlocking, FCC, and IMEI check resource
    device_status_urls_node = evaluator.add_parallel(
        id="URLs_For_Device_Status",
        desc="Provides URL(s) supporting device-status requirements (unlocked/unlock-eligible and FCC certification), and provides a URL or official resource for checking/confirming IMEI status (since IMEI cleanliness is unit-specific).",
        parent=node,
        critical=True,
    )

    unlock_urls_leaf = evaluator.add_leaf(
        id="URLs_For_Unlocking",
        desc="Provides URL(s) supporting sold-unlocked or unlock-eligible status.",
        parent=device_status_urls_node,
        critical=True,
    )
    await evaluator.verify(
        claim="These URLs substantiate that the device is sold unlocked in the US or is unlock-eligible under US policies.",
        node=unlock_urls_leaf,
        sources=data.unlock_or_unlock_eligible_urls if _nonempty(data.unlock_or_unlock_eligible_urls) else None,
        additional_instruction="Accept official product/retailer or carrier policy documentation tied to this model.",
    )

    fcc_urls_leaf = evaluator.add_leaf(
        id="URLs_For_FCC",
        desc="Provides URL(s) supporting FCC certification for US operation.",
        parent=device_status_urls_node,
        critical=True,
    )
    await evaluator.verify(
        claim="These URLs substantiate US FCC certification/approval for this device or its US variant (FCC ID).",
        node=fcc_urls_leaf,
        sources=data.fcc_cert_urls if _nonempty(data.fcc_cert_urls) else None,
        additional_instruction="Prefer the FCC ID search database page for the exact model or an authoritative manufacturer link referencing the FCC ID.",
    )

    imei_urls_leaf = evaluator.add_leaf(
        id="URLs_For_IMEI_Check",
        desc="Provides URL(s) to official IMEI check resources to confirm clean/not-blacklisted status.",
        parent=device_status_urls_node,
        critical=True,
    )
    await evaluator.verify(
        claim="These URLs are official resources for checking an IMEI's blacklist/clean status.",
        node=imei_urls_leaf,
        sources=data.imei_check_urls if _nonempty(data.imei_check_urls) else None,
        additional_instruction="Accept carrier or widely-recognized industry resources for IMEI status checks.",
    )

    # Coverage URLs: ensure provided and state-level applicability
    coverage_urls_node = evaluator.add_parallel(
        id="URLs_For_Geographic_Coverage",
        desc="Provides URL(s) (e.g., carrier coverage map/resources) used to support the coverage discussion for Texas, Georgia, New York, and California.",
        parent=node,
        critical=True,
    )

    cov_exist = evaluator.add_custom_node(
        result=_nonempty(data.coverage_urls),
        id="Coverage_URLs_Provided",
        desc="Provides at least one coverage map/resource URL.",
        parent=coverage_urls_node,
        critical=True,
    )

    # Create one leaf per state, using the same coverage URLs
    for state in REQUIRED_STATES:
        state_id = f"Coverage_URLs_{state.replace(' ', '_')}"
        state_leaf = evaluator.add_leaf(
            id=state_id,
            desc=f"Coverage resource(s) usable to determine service availability in {state}.",
            parent=coverage_urls_node,
            critical=True,
        )
        await evaluator.verify(
            claim=f"These URLs are official coverage map/resources that can be used to check service availability in {state}.",
            node=state_leaf,
            sources=data.coverage_urls if _nonempty(data.coverage_urls) else None,
            additional_instruction=(
                "Accept official carrier coverage map pages or equivalent authoritative coverage resources; "
                "the page need not display the state name inline if it is an interactive national map for checking state/local coverage."
            ),
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
    """
    Evaluate a single answer for the 2026 smartphone tri-carrier reliability task and return a structured result.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall: independent categories; critical gating handled in child node
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

    # Record "ground truth" requirement lists for clarity in the summary
    evaluator.add_ground_truth(
        {
            "required_lte_bands": REQUIRED_LTE_BANDS,
            "required_nr_bands": REQUIRED_NR_BANDS,
            "required_states": REQUIRED_STATES,
            "required_carriers": REQUIRED_CARRIERS,
        },
        gt_type="requirements",
    )

    # 1) Extract structured information from the answer
    data: SmartphoneExtraction = await evaluator.extract(
        prompt=prompt_extract_smartphone(),
        template_class=SmartphoneExtraction,
        extraction_name="smartphone_extraction",
    )

    # 2) Build the critical task root (as a child of the framework root)
    task_root = evaluator.add_parallel(
        id="Smartphone_Model_Meets_All_Stated_Requirements",
        desc="Answer identifies one US-available (in 2026) smartphone model and shows it meets all stated band, tri-carrier feature, dual-SIM, device-status, and state-coverage requirements, including providing supporting documentation URLs for each requirement category.",
        parent=root,
        critical=True,
    )

    # 3) Verification subtrees
    await verify_device_identification_and_availability(evaluator, task_root, data)
    await verify_radio_access_bands(evaluator, task_root, data)
    await verify_tri_carrier_features(evaluator, task_root, data)
    await verify_device_features_and_status(evaluator, task_root, data)
    await verify_geographic_coverage(evaluator, task_root, data)
    await verify_supporting_documentation_urls(evaluator, task_root, data)

    # 4) Return structured result
    return evaluator.get_summary()