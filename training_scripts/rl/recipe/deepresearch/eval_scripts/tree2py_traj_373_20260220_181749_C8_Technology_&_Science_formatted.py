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
TASK_ID = "us_device_2025_compliance"
TASK_DESCRIPTION = (
    "Identify a mobile wireless device model that was launched or received a major certification update in 2025 and "
    "meets all of the specified US market requirements (FCC authorization, SAR compliance, band support, carrier "
    "certification, HAC, battery safety, EMC compliance, and device category). Provide manufacturer and model, FCC ID, "
    "SAR value, supported bands, carrier certification, HAC ratings, battery safety standard, EMC documentation, and "
    "device category. All claims must be supported by official documentation URLs."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DeviceBasics(BaseModel):
    manufacturer: Optional[str] = None
    model_name: Optional[str] = None
    device_category: Optional[str] = None  # smartphone / IoT module / mobile hotspot / tablet / wearable
    launch_or_update_year: Optional[str] = None  # e.g., "2025", "Launched Jan 2025"
    manufacturer_contact_url: Optional[str] = None  # official contact/support page if provided


class FCCInfo(BaseModel):
    fcc_id: Optional[str] = None
    fcc_id_urls: List[str] = Field(default_factory=list)  # official OET/EAS URLs preferred


class SARInfo(BaseModel):
    sar_value_wkg: Optional[str] = None  # e.g., "1.20 W/kg"
    sar_doc_urls: List[str] = Field(default_factory=list)  # official SAR doc or FCC filing URLs


class BandInfo(BaseModel):
    cellular_bands: List[str] = Field(default_factory=list)  # as stated; can include LTE/NR bands
    band_spec_urls: List[str] = Field(default_factory=list)  # official spec/manufacturer pages


class CarrierInfo(BaseModel):
    carrier_name: Optional[str] = None  # AT&T / Verizon / T-Mobile
    carrier_doc_urls: List[str] = Field(default_factory=list)  # official carrier listing/approval pages


class PTCRBInfo(BaseModel):
    ptcrb_status: Optional[str] = None  # e.g., "PTCRB Certified", "Yes", "No"
    ptcrb_doc_urls: List[str] = Field(default_factory=list)  # ptcrb.org or official documentation


class HACInfo(BaseModel):
    hac_m_rating: Optional[str] = None  # M3/M4
    hac_t_rating: Optional[str] = None  # T3/T4
    hac_doc_urls: List[str] = Field(default_factory=list)  # official HAC docs or FCC filings


class BatteryInfo(BaseModel):
    battery_standard: Optional[str] = None  # UL 2054 / IEC 62133-2 / UL 2595
    battery_doc_urls: List[str] = Field(default_factory=list)  # official certification docs


class RoHSInfo(BaseModel):
    rohs_statement: Optional[str] = None  # textual statement if provided
    rohs_doc_urls: List[str] = Field(default_factory=list)  # official manufacturer RoHS pages


class EMCInfo(BaseModel):
    emc_standard: Optional[str] = None  # e.g., "FCC Part 15", "EN 301 489", etc.
    emc_doc_urls: List[str] = Field(default_factory=list)  # official EMC compliance docs


class DeviceExtraction(BaseModel):
    basics: Optional[DeviceBasics] = None
    fcc: Optional[FCCInfo] = None
    sar: Optional[SARInfo] = None
    bands: Optional[BandInfo] = None
    carrier: Optional[CarrierInfo] = None
    ptcrb: Optional[PTCRBInfo] = None
    hac: Optional[HACInfo] = None
    battery: Optional[BatteryInfo] = None
    rohs: Optional[RoHSInfo] = None
    emc: Optional[EMCInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_device() -> str:
    return """
    Extract the complete device compliance and specification information stated in the answer. 
    Return a JSON object matching this schema:

    {
      "basics": {
        "manufacturer": string or null,
        "model_name": string or null,
        "device_category": string or null,  // one of: smartphone, IoT module, mobile hotspot, tablet, wearable
        "launch_or_update_year": string or null, // year or phrase that includes 2025 if mentioned
        "manufacturer_contact_url": string or null // official contact/support URL if provided
      },
      "fcc": {
        "fcc_id": string or null,
        "fcc_id_urls": [array of URLs] // prefer official FCC OET/EAS pages; include all URLs cited for FCC ID
      },
      "sar": {
        "sar_value_wkg": string or null, // e.g., "1.20 W/kg" exactly as written
        "sar_doc_urls": [array of URLs] // official SAR documentation or FCC filings URLs cited
      },
      "bands": {
        "cellular_bands": [array of strings], // LTE/NR bands as written (e.g., "LTE Band 2", "NR n66")
        "band_spec_urls": [array of URLs] // official spec pages listing supported bands
      },
      "carrier": {
        "carrier_name": string or null, // AT&T, Verizon, or T-Mobile
        "carrier_doc_urls": [array of URLs] // official carrier listing/approval URLs
      },
      "ptcrb": {
        "ptcrb_status": string or null, // e.g., "Certified", "Yes", "No"
        "ptcrb_doc_urls": [array of URLs] // official PTCRB database or certificate URLs
      },
      "hac": {
        "hac_m_rating": string or null, // e.g., "M3", "M4"
        "hac_t_rating": string or null, // e.g., "T3", "T4"
        "hac_doc_urls": [array of URLs] // official HAC rating documentation URLs
      },
      "battery": {
        "battery_standard": string or null, // UL 2054, IEC 62133-2, or UL 2595
        "battery_doc_urls": [array of URLs] // official certification or compliance URLs
      },
      "rohs": {
        "rohs_statement": string or null, // textual statement of compliance if provided
        "rohs_doc_urls": [array of URLs] // official RoHS compliance URLs
      },
      "emc": {
        "emc_standard": string or null, // e.g., "FCC Part 15" or equivalent
        "emc_doc_urls": [array of URLs] // official EMC compliance documentation URLs
      }
    }

    Rules:
    - Extract only what is explicitly present in the answer. Do not invent values or URLs.
    - For each URL field, extract actual URLs shown in the answer text. Prefer official domains: manufacturers, fcc.gov, carrier sites, ptcrb.org, test labs.
    - If a field is missing, set it to null or an empty array accordingly.
    - Keep band names exactly as presented (e.g., "LTE Band 2", "Band 4 AWS", "NR n66").
    - Do not normalize numeric values; keep formatting as in the answer (e.g., "1.20 W/kg" vs "1.2 W/kg").
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _list_clean(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    cleaned = []
    for u in urls:
        if not u:
            continue
        v = u.strip()
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            cleaned.append(v)
    return cleaned


def combine_sources(*args: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    for arr in args:
        combined.extend(_list_clean(arr))
    # dedup
    deduped = []
    seen = set()
    for u in combined:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def text_or_empty(v: Optional[str]) -> str:
    return (v or "").strip()


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_device_identification(evaluator: Evaluator, parent, ex: DeviceExtraction) -> None:
    basics = ex.basics or DeviceBasics()
    manufacturer = text_or_empty(basics.manufacturer)
    model = text_or_empty(basics.model_name)

    node = evaluator.add_sequential(
        id="device_identification",
        desc="Correctly identify a specific device model with manufacturer name and model number",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(manufacturer) and bool(model),
        id="device_identification_provided",
        desc="Manufacturer and model name are provided",
        parent=node,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="device_identification_verified",
        desc="Manufacturer and model are supported by official documentation",
        parent=node,
        critical=True
    )

    sources = combine_sources(
        (ex.bands or BandInfo()).band_spec_urls,
        (ex.fcc or FCCInfo()).fcc_id_urls,
        (ex.carrier or CarrierInfo()).carrier_doc_urls,
        (ex.emc or EMCInfo()).emc_doc_urls,
        (ex.battery or BatteryInfo()).battery_doc_urls,
        (ex.hac or HACInfo()).hac_doc_urls,
        (ex.sar or SARInfo()).sar_doc_urls,
    )
    claim = f"An official documentation page shows the device model '{model}' manufactured by '{manufacturer}'."
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=sources,
        additional_instruction="The documentation should mention both the manufacturer and the exact model name. Prefer manufacturer spec pages or FCC filings; accept carrier certification pages if they clearly reference the model."
    )


async def build_launch_year(evaluator: Evaluator, parent, ex: DeviceExtraction) -> None:
    basics = ex.basics or DeviceBasics()
    year_text = text_or_empty(basics.launch_or_update_year)

    node = evaluator.add_sequential(
        id="launch_year_2025",
        desc="Verify the device was launched or received a major update/certification in 2025",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(year_text) and ("2025" in year_text),
        id="launch_year_2025_provided",
        desc="Launch/update/certification year (2025) is provided",
        parent=node,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="launch_year_2025_verified",
        desc="Official documentation shows launch or major certification update in 2025",
        parent=node,
        critical=True
    )

    sources = combine_sources(
        (ex.sar or SARInfo()).sar_doc_urls,
        (ex.fcc or FCCInfo()).fcc_id_urls,
        (ex.carrier or CarrierInfo()).carrier_doc_urls,
        (ex.bands or BandInfo()).band_spec_urls,
        (ex.ptcrb or PTCRBInfo()).ptcrb_doc_urls
    )
    claim = "The device was launched or received a major certification update in 2025."
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=sources,
        additional_instruction="Accept official press releases, manufacturer spec pages with dates, FCC grant/application dates, PTCRB certification dates, or carrier certification dates that fall in 2025."
    )


async def build_fcc_authorization(evaluator: Evaluator, parent, ex: DeviceExtraction) -> None:
    fcc = ex.fcc or FCCInfo()
    fcc_id = text_or_empty(fcc.fcc_id)
    fcc_sources = _list_clean(fcc.fcc_id_urls)

    node = evaluator.add_sequential(
        id="fcc_authorization",
        desc="Device has valid FCC equipment authorization with assigned FCC ID",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(fcc_id),
        id="fcc_id_provided",
        desc="FCC ID number is provided",
        parent=node,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="fcc_id_verification",
        desc="URL reference to FCC ID database or official documentation verifying the FCC ID",
        parent=node,
        critical=True
    )

    claim = f"The FCC ID '{fcc_id}' appears in the official FCC equipment authorization database and corresponds to this device."
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=fcc_sources,
        additional_instruction="Prefer fcc.gov OET/EAS pages. The page should show a grant/application for the FCC ID and mention the manufacturer and/or model."
    )


async def build_sar_compliance(evaluator: Evaluator, parent, ex: DeviceExtraction) -> None:
    sar = ex.sar or SARInfo()
    sar_val = text_or_empty(sar.sar_value_wkg)
    sar_sources = _list_clean(sar.sar_doc_urls)

    node = evaluator.add_sequential(
        id="sar_compliance",
        desc="Device meets FCC SAR limit of 1.6 W/kg",
        parent=parent,
        critical=True
    )

    value_node = evaluator.add_leaf(
        id="sar_value_reported",
        desc="SAR test value is reported and does not exceed 1.6 W/kg",
        parent=node,
        critical=True
    )
    value_claim = (
        f"The device's SAR maximum (averaged over 1 gram) is {sar_val} and does not exceed 1.6 W/kg."
        if sar_val else
        "The device's SAR maximum (averaged over 1 gram) does not exceed 1.6 W/kg."
    )
    await evaluator.verify(
        claim=value_claim,
        node=value_node,
        sources=sar_sources,
        additional_instruction="Check official SAR documentation or FCC filings. Accept head or body 1g measurements; confirm all reported values are ≤ 1.6 W/kg."
    )

    verify_node = evaluator.add_leaf(
        id="sar_verification",
        desc="URL reference to official SAR documentation or FCC filing",
        parent=node,
        critical=True
    )
    verify_claim = "Official documentation demonstrates SAR compliance at or below 1.6 W/kg averaged over 1g tissue."
    await evaluator.verify(
        claim=verify_claim,
        node=verify_node,
        sources=sar_sources if sar_sources else (ex.fcc or FCCInfo()).fcc_id_urls,
        additional_instruction="Verify the documentation explicitly lists SAR values and demonstrates compliance to the FCC limit."
    )


async def build_band_support(evaluator: Evaluator, parent, ex: DeviceExtraction) -> None:
    bands = ex.bands or BandInfo()
    band_sources = _list_clean(bands.band_spec_urls)
    band_list_str = ", ".join(bands.cellular_bands) if bands.cellular_bands else "the bands listed on the spec page"

    node = evaluator.add_parallel(
        id="cellular_band_support",
        desc="Device supports required LTE or 5G bands for US carriers",
        parent=parent,
        critical=True
    )

    # Band 2
    b2_node = evaluator.add_leaf(
        id="band_2_support",
        desc="Device supports LTE Band 2 (1900 MHz) or 5G equivalent",
        parent=node,
        critical=True
    )
    b2_claim = (
        f"The device supports LTE Band 2 (1900 MHz PCS) or 5G NR n2, as indicated among supported bands ({band_list_str})."
    )
    await evaluator.verify(
        claim=b2_claim,
        node=b2_node,
        sources=band_sources if band_sources else (ex.fcc or FCCInfo()).fcc_id_urls,
        additional_instruction="Accept LTE Band 2 or 5G NR n2 as equivalent PCS support for US carriers."
    )

    # Band 4
    b4_node = evaluator.add_leaf(
        id="band_4_support",
        desc="Device supports LTE Band 4 (1700/2100 MHz AWS) or 5G equivalent",
        parent=node,
        critical=True
    )
    b4_claim = (
        f"The device supports LTE Band 4 (AWS-1) or 5G NR bands covering AWS (e.g., n4/n66), as indicated among supported bands ({band_list_str})."
    )
    await evaluator.verify(
        claim=b4_claim,
        node=b4_node,
        sources=band_sources if band_sources else (ex.fcc or FCCInfo()).fcc_id_urls,
        additional_instruction="Accept LTE Band 4 or 5G NR n4/n66 as equivalent AWS support."
    )

    # Band 12
    b12_node = evaluator.add_leaf(
        id="band_12_support",
        desc="Device supports LTE Band 12 (700 MHz) or 5G equivalent",
        parent=node,
        critical=True
    )
    b12_claim = (
        f"The device supports LTE Band 12 (700 MHz) or a 5G NR band covering the 700 MHz block (e.g., n12), as indicated among supported bands ({band_list_str})."
    )
    await evaluator.verify(
        claim=b12_claim,
        node=b12_node,
        sources=band_sources if band_sources else (ex.fcc or FCCInfo()).fcc_id_urls,
        additional_instruction="Accept LTE Band 12 or 5G NR n12 as equivalent 700 MHz support."
    )

    # Band specification verification
    spec_node = evaluator.add_leaf(
        id="band_specification_verification",
        desc="URL reference to official device specifications listing supported bands",
        parent=node,
        critical=True
    )
    spec_claim = "The official device specification page lists supported cellular bands."
    await evaluator.verify(
        claim=spec_claim,
        node=spec_node,
        sources=band_sources,
        additional_instruction="This should be an official manufacturer spec page or equivalent authoritative documentation listing the supported bands."
    )


async def build_carrier_certification(evaluator: Evaluator, parent, ex: DeviceExtraction) -> None:
    carrier = ex.carrier or CarrierInfo()
    carrier_name = text_or_empty(carrier.carrier_name)
    carrier_sources = _list_clean(carrier.carrier_doc_urls)

    node = evaluator.add_sequential(
        id="carrier_certification",
        desc="Device has certification from at least one major US carrier (AT&T, Verizon, or T-Mobile)",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=carrier_name.lower() in {"at&t", "att", "verizon", "t-mobile", "t mobile"} if carrier_name else False,
        id="carrier_name",
        desc="Name of the certifying carrier is provided",
        parent=node,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="carrier_certification_verification",
        desc="URL reference to carrier's certified devices list or official approval documentation",
        parent=node,
        critical=True
    )

    claim = f"The device is officially certified/approved by {carrier_name} for use on their network."
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=carrier_sources,
        additional_instruction="Use official carrier device listing or certification pages (AT&T, Verizon, T-Mobile). Third-party blogs are not acceptable."
    )


async def build_ptcrb_certification(evaluator: Evaluator, parent, ex: DeviceExtraction) -> None:
    pt = ex.ptcrb or PTCRBInfo()
    status = text_or_empty(pt.ptcrb_status)
    pt_sources = _list_clean(pt.ptcrb_doc_urls)

    node = evaluator.add_sequential(
        id="ptcrb_certification",
        desc="Device has PTCRB certification for North American cellular networks",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(status) and ("cert" in status.lower() or "yes" in status.lower()),
        id="ptcrb_status",
        desc="PTCRB certification status is confirmed",
        parent=node,
        critical=False
    )

    verify_node = evaluator.add_leaf(
        id="ptcrb_verification",
        desc="URL reference to PTCRB certified devices database or official documentation",
        parent=node,
        critical=False
    )
    claim = "The device is PTCRB certified according to official documentation."
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=pt_sources,
        additional_instruction="Prefer ptcrb.org database pages or official certificates from recognized test labs referencing PTCRB certification."
    )


async def build_hac(evaluator: Evaluator, parent, ex: DeviceExtraction) -> None:
    hac = ex.hac or HACInfo()
    m_rating = text_or_empty(hac.hac_m_rating)
    t_rating = text_or_empty(hac.hac_t_rating)
    hac_sources = _list_clean(hac.hac_doc_urls)

    node = evaluator.add_parallel(
        id="hearing_aid_compatibility",
        desc="Device meets FCC hearing aid compatibility requirements (M3/T3 or better)",
        parent=parent,
        critical=True
    )

    m_node = evaluator.add_leaf(
        id="m_rating",
        desc="Device has M3 or M4 rating for acoustic coupling",
        parent=node,
        critical=True
    )
    m_claim = (
        f"The device has HAC M-rating {m_rating}, which is M3 or M4."
        if m_rating else
        "The device has HAC M-rating of M3 or M4."
    )
    await evaluator.verify(
        claim=m_claim,
        node=m_node,
        sources=hac_sources,
        additional_instruction="Confirm from official HAC documentation or FCC filings that the M-rating is M3 or M4."
    )

    t_node = evaluator.add_leaf(
        id="t_rating",
        desc="Device has T3 or T4 rating for inductive coupling",
        parent=node,
        critical=True
    )
    t_claim = (
        f"The device has HAC T-rating {t_rating}, which is T3 or T4."
        if t_rating else
        "The device has HAC T-rating of T3 or T4."
    )
    await evaluator.verify(
        claim=t_claim,
        node=t_node,
        sources=hac_sources,
        additional_instruction="Confirm from official HAC documentation or FCC filings that the T-rating is T3 or T4."
    )

    ver_node = evaluator.add_leaf(
        id="hac_verification",
        desc="URL reference to official HAC rating documentation",
        parent=node,
        critical=True
    )
    ver_claim = "Official documentation lists HAC ratings and confirms the device meets or exceeds M3/T3."
    await evaluator.verify(
        claim=ver_claim,
        node=ver_node,
        sources=hac_sources,
        additional_instruction="Prefer manufacturer HAC pages or FCC filings where HAC ratings are explicitly stated."
    )


async def build_battery_safety(evaluator: Evaluator, parent, ex: DeviceExtraction) -> None:
    bat = ex.battery or BatteryInfo()
    std = text_or_empty(bat.battery_standard)
    bat_sources = _list_clean(bat.battery_doc_urls)

    node = evaluator.add_sequential(
        id="battery_safety_certification",
        desc="Device's battery complies with recognized safety standards (UL 2054, IEC 62133, or UL 2595)",
        parent=parent,
        critical=True
    )

    std_node = evaluator.add_leaf(
        id="battery_standard",
        desc="Specific battery safety standard compliance is identified (UL 2054, IEC 62133-2, or UL 2595)",
        parent=node,
        critical=True
    )
    std_claim = (
        f"The device's battery is certified compliant with {std}."
        if std else
        "The device's battery is certified compliant with UL 2054, IEC 62133-2, or UL 2595."
    )
    await evaluator.verify(
        claim=std_claim,
        node=std_node,
        sources=bat_sources,
        additional_instruction="Confirm from official certification or manufacturer safety compliance statements. The standard must be one of UL 2054, IEC 62133-2, or UL 2595."
    )

    ver_node = evaluator.add_leaf(
        id="battery_certification_verification",
        desc="URL reference to battery certification documentation or official safety compliance statement",
        parent=node,
        critical=True
    )
    ver_claim = "Official documentation confirms battery safety compliance under the stated standard."
    await evaluator.verify(
        claim=ver_claim,
        node=ver_node,
        sources=bat_sources,
        additional_instruction="Prefer certification documents from recognized labs or official manufacturer compliance statements referencing the specific standard."
    )


async def build_rohs(evaluator: Evaluator, parent, ex: DeviceExtraction) -> None:
    rohs = ex.rohs or RoHSInfo()
    rohs_sources = _list_clean(rohs.rohs_doc_urls)

    node = evaluator.add_sequential(
        id="rohs_compliance",
        desc="Device complies with RoHS directive restricting hazardous substances",
        parent=parent,
        critical=False
    )

    stmt_node = evaluator.add_leaf(
        id="rohs_statement",
        desc="RoHS compliance is stated or documented",
        parent=node,
        critical=False
    )
    stmt_claim = "The device is stated to be RoHS compliant in official manufacturer documentation."
    await evaluator.verify(
        claim=stmt_claim,
        node=stmt_node,
        sources=rohs_sources,
        additional_instruction="Confirm from official manufacturer compliance pages or documentation that RoHS compliance is claimed."
    )

    ver_node = evaluator.add_leaf(
        id="rohs_verification",
        desc="URL reference to RoHS compliance documentation or manufacturer statement",
        parent=node,
        critical=False
    )
    ver_claim = "Official documentation confirms RoHS compliance for the device."
    await evaluator.verify(
        claim=ver_claim,
        node=ver_node,
        sources=rohs_sources,
        additional_instruction="Use manufacturer compliance documentation or official statements."
    )


async def build_emc(evaluator: Evaluator, parent, ex: DeviceExtraction) -> None:
    emc = ex.emc or EMCInfo()
    std = text_or_empty(emc.emc_standard)
    emc_sources = _list_clean(emc.emc_doc_urls)

    node = evaluator.add_sequential(
        id="emc_compliance",
        desc="Device meets electromagnetic compatibility requirements",
        parent=parent,
        critical=True
    )

    std_node = evaluator.add_leaf(
        id="emc_standard",
        desc="Compliance with EMC standards (FCC Part 15 or equivalent) is documented",
        parent=node,
        critical=True
    )
    std_claim = (
        f"The device's EMC compliance is documented for {std}."
        if std else
        "The device's EMC compliance is documented (e.g., FCC Part 15 or equivalent)."
    )
    await evaluator.verify(
        claim=std_claim,
        node=std_node,
        sources=emc_sources if emc_sources else (ex.fcc or FCCInfo()).fcc_id_urls,
        additional_instruction="Confirm from official EMC test reports or FCC filings that the device meets EMC requirements (e.g., FCC Part 15)."
    )

    ver_node = evaluator.add_leaf(
        id="emc_verification",
        desc="URL reference to EMC compliance documentation",
        parent=node,
        critical=True
    )
    ver_claim = "Official documentation confirms EMC compliance for the device."
    await evaluator.verify(
        claim=ver_claim,
        node=ver_node,
        sources=emc_sources if emc_sources else (ex.fcc or FCCInfo()).fcc_id_urls,
        additional_instruction="Prefer EMC test reports or FCC filings evidencing compliance."
    )


async def build_device_category(evaluator: Evaluator, parent, ex: DeviceExtraction) -> None:
    basics = ex.basics or DeviceBasics()
    category = text_or_empty(basics.device_category)
    cat_node = evaluator.add_leaf(
        id="device_category",
        desc="Device category is clearly identified (smartphone, IoT module, mobile hotspot, tablet, or wearable)",
        parent=parent,
        critical=True
    )
    claim = (
        f"The device is a '{category}', which is one of: smartphone, IoT cellular module, mobile hotspot, tablet, or wearable."
        if category else
        "The device category is one of: smartphone, IoT cellular module, mobile hotspot, tablet, or wearable."
    )
    sources = combine_sources(
        (ex.bands or BandInfo()).band_spec_urls,
        (ex.fcc or FCCInfo()).fcc_id_urls
    )
    await evaluator.verify(
        claim=claim,
        node=cat_node,
        sources=sources,
        additional_instruction="Check the official spec or FCC documentation to confirm the device category."
    )


async def build_manufacturer_info(evaluator: Evaluator, parent, ex: DeviceExtraction) -> None:
    basics = ex.basics or DeviceBasics()
    manufacturer = text_or_empty(basics.manufacturer)
    contact_url = text_or_empty(basics.manufacturer_contact_url)

    evaluator.add_custom_node(
        result=bool(manufacturer) and bool(contact_url),
        id="manufacturer_information",
        desc="Manufacturer name and contact information are provided",
        parent=parent,
        critical=False
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 2025 US device compliance task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation at root
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

    # NOTE: Root is set to non-critical to allow a mix of critical and non-critical children.
    #       Individual nodes below carry correct criticality. This preserves gating:
    #       any failed critical child will gate parent aggregation to 0.0 automatically.

    # Extract structured device information
    ex: DeviceExtraction = await evaluator.extract(
        prompt=prompt_extract_device(),
        template_class=DeviceExtraction,
        extraction_name="device_info"
    )

    # Build verification tree according to rubric
    # Device identification (critical)
    await build_device_identification(evaluator, root, ex)

    # Launch/update year (critical - must be 2025)
    await build_launch_year(evaluator, root, ex)

    # FCC authorization (critical)
    await build_fcc_authorization(evaluator, root, ex)

    # SAR compliance (critical)
    await build_sar_compliance(evaluator, root, ex)

    # Cellular band support (critical)
    await build_band_support(evaluator, root, ex)

    # Carrier certification (critical)
    await build_carrier_certification(evaluator, root, ex)

    # PTCRB certification (non-critical, partial credit)
    await build_ptcrb_certification(evaluator, root, ex)

    # Hearing Aid Compatibility (critical)
    await build_hac(evaluator, root, ex)

    # Battery safety certification (critical)
    await build_battery_safety(evaluator, root, ex)

    # RoHS compliance (non-critical, partial credit)
    await build_rohs(evaluator, root, ex)

    # EMC compliance (critical)
    await build_emc(evaluator, root, ex)

    # Device category (critical)
    await build_device_category(evaluator, root, ex)

    # Manufacturer information (non-critical)
    await build_manufacturer_info(evaluator, root, ex)

    # Return structured evaluation summary
    return evaluator.get_summary()