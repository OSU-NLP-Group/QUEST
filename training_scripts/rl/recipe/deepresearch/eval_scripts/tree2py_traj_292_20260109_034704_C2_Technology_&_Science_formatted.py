import asyncio
import logging
from datetime import datetime, date
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fda_ai_ecg_lvef_q3_2025"
TASK_DESCRIPTION = (
    "Identify two FDA-cleared AI-enabled medical devices from the third quarter of 2025 "
    "(July 1 - September 30, 2025) that meet all of the following criteria: "
    "(1) Classified under the Cardiovascular specialty by the FDA; "
    "(2) Utilize artificial intelligence or machine learning to analyze electrocardiogram (ECG) data; "
    "(3) Specifically intended for detecting, identifying, or screening for low left ventricular ejection fraction (LVEF) "
    "or related cardiac dysfunction. For each device, provide: the official device name as listed in FDA records, "
    "the manufacturer's name, the FDA 510(k) premarket notification number, the FDA clearance decision date, "
    "a brief description of the device's clinical indication related to LVEF detection, and a direct URL link to the device's "
    "entry in the FDA's 510(k) Premarket Notification Database."
)

Q3_2025_START = date(2025, 7, 1)
Q3_2025_END = date(2025, 9, 30)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DeviceInfo(BaseModel):
    """Structured fields for one FDA-cleared device."""
    device_name: Optional[str] = None
    manufacturer: Optional[str] = None
    k_number: Optional[str] = None  # FDA 510(k) number, typically like "Kxxxxxx"
    clearance_date: Optional[str] = None  # String as presented in the answer
    lvef_indication_desc: Optional[str] = None  # Brief description related to LVEF
    fda_url: Optional[str] = None  # Direct URL to FDA 510(k) database entry


class DevicesExtraction(BaseModel):
    """Extraction of devices listed in the answer."""
    devices: List[DeviceInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_devices() -> str:
    return (
        "From the provided answer, extract all AI-enabled medical devices that the answer claims meet the criteria. "
        "For each device, return an object with the following fields strictly as presented in the answer text:\n"
        "1) device_name: Official device name as listed in FDA records (string)\n"
        "2) manufacturer: Manufacturer/applicant name (string)\n"
        "3) k_number: FDA 510(k) premarket notification number (string, e.g., 'K123456')\n"
        "4) clearance_date: FDA clearance decision date (string, e.g., 'September 15, 2025' or '2025-09-15')\n"
        "5) lvef_indication_desc: A brief description of the device’s clinical indication related to LVEF detection (string)\n"
        "6) fda_url: Direct URL to the device’s entry in the FDA 510(k) Premarket Notification Database (string URL)\n\n"
        "Return a JSON object with an array field 'devices'. If the answer includes more than two devices, include all of them; "
        "downstream evaluation will consider only the first two. If a field is not present in the answer, use null."
    )


# --------------------------------------------------------------------------- #
# Helpers for date parsing and constraints                                    #
# --------------------------------------------------------------------------- #
def _normalize_ordinal_suffixes(s: str) -> str:
    """Remove ordinal suffixes like 'st', 'nd', 'rd', 'th' from day component."""
    # Example: "September 1st, 2025" -> "September 1, 2025"
    for suf in ("st", "nd", "rd", "th"):
        s = s.replace(f" {suf},", ",")  # already before comma (rare)
        s = s.replace(f"{suf},", ",")
        s = s.replace(f" {suf} ", " ")
    return s


def parse_date_str(date_str: Optional[str]) -> Optional[date]:
    """Try multiple formats to parse a date string into a date object."""
    if not date_str or not isinstance(date_str, str):
        return None
    s = date_str.strip()
    if not s:
        return None

    s = _normalize_ordinal_suffixes(s)

    fmts = [
        "%B %d, %Y",   # September 15, 2025
        "%b %d, %Y",   # Sep 15, 2025
        "%Y-%m-%d",    # 2025-09-15
        "%m/%d/%Y",    # 09/15/2025
        "%Y/%m/%d",    # 2025/09/15
        "%d %B %Y",    # 15 September 2025
        "%d %b %Y",    # 15 Sep 2025
    ]

    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt).date()
            return dt
        except Exception:
            continue

    # Try basic cleanup cases (remove commas)
    try:
        s2 = s.replace(",", " ")
        for fmt in ["%B %d %Y", "%b %d %Y"]:
            try:
                dt = datetime.strptime(s2, fmt).date()
                return dt
            except Exception:
                continue
    except Exception:
        pass

    return None


def is_in_q3_2025(date_str: Optional[str]) -> bool:
    """Check whether the given date string is within Q3 2025 inclusive."""
    d = parse_date_str(date_str)
    if d is None:
        return False
    return Q3_2025_START <= d <= Q3_2025_END


# --------------------------------------------------------------------------- #
# Verification logic for a single device                                      #
# --------------------------------------------------------------------------- #
async def verify_device(
    evaluator: Evaluator,
    parent_node,
    dev: DeviceInfo,
    ordinal_index: int,
) -> None:
    """
    Build verification subtree for one device.
    All leaves under the device node are critical as the device must meet all criteria.
    """

    device_display_idx = ordinal_index + 1
    dev_node = evaluator.add_parallel(
        id=f"device_{device_display_idx}",
        desc=f"Device {device_display_idx} (should meet all qualifying criteria and include all required fields)",
        parent=parent_node,
        critical=False,  # The device subtree contributes partial credit to root
    )

    # 1) Field existence checks (critical)
    evaluator.add_custom_node(
        result=bool(dev.device_name and dev.device_name.strip()),
        id=f"device_{device_display_idx}_name",
        desc="Provide the official device name as listed in FDA records",
        parent=dev_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(dev.manufacturer and dev.manufacturer.strip()),
        id=f"device_{device_display_idx}_manufacturer",
        desc="Provide the manufacturer's name as listed/identifiable in FDA records",
        parent=dev_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(dev.k_number and dev.k_number.strip()),
        id=f"device_{device_display_idx}_510k_number",
        desc="Provide the FDA 510(k) premarket notification number for the device",
        parent=dev_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(dev.clearance_date and dev.clearance_date.strip()),
        id=f"device_{device_display_idx}_clearance_decision_date",
        desc="Provide the FDA clearance decision date for the device",
        parent=dev_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=is_in_q3_2025(dev.clearance_date),
        id=f"device_{device_display_idx}_q3_2025_date_constraint",
        desc="The clearance decision date falls between July 1, 2025 and September 30, 2025 (inclusive)",
        parent=dev_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(dev.lvef_indication_desc and dev.lvef_indication_desc.strip()),
        id=f"device_{device_display_idx}_indication_description_provided",
        desc="Provide a brief description of the device’s clinical indication related to LVEF detection",
        parent=dev_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(dev.fda_url and dev.fda_url.strip()),
        id=f"device_{device_display_idx}_reference_url",
        desc="Provide a direct URL to the device’s entry in the FDA 510(k) Premarket Notification Database (or other FDA official database explicitly allowed by constraints)",
        parent=dev_node,
        critical=True,
    )

    # 2) Constraints verification by URL (critical leaves)
    cardio_node = evaluator.add_leaf(
        id=f"device_{device_display_idx}_cardiovascular_specialty_constraint",
        desc="The device is classified under the Cardiovascular specialty category by the FDA",
        parent=dev_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This FDA device entry indicates the medical specialty classification is Cardiovascular.",
        node=cardio_node,
        sources=dev.fda_url,
        additional_instruction=(
            "On the FDA 510(k) page, look for fields like 'Regulation Medical Specialty' or "
            "'Assigned Medical Specialty' showing 'Cardiovascular'. Allow minor variants of wording."
        ),
    )

    ai_ecg_node = evaluator.add_leaf(
        id=f"device_{device_display_idx}_ai_ml_ecg_constraint",
        desc="The device uses artificial intelligence or machine learning to analyze electrocardiogram (ECG) data",
        parent=dev_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This device utilizes artificial intelligence or machine learning to analyze ECG (electrocardiogram) data.",
        node=ai_ecg_node,
        sources=dev.fda_url,
        additional_instruction=(
            "Check for mentions of 'AI', 'artificial intelligence', 'machine learning', 'deep learning', or similar, "
            "in relation to ECG signal/waveform analysis (e.g., 12-lead ECG). Evidence may appear in 'Device Description', "
            "'Indications for Use', or 'Technology' sections."
        ),
    )

    lvef_node = evaluator.add_leaf(
        id=f"device_{device_display_idx}_lvef_indication_constraint",
        desc="The device is specifically intended for detecting, identifying, or screening for low LVEF or related cardiac dysfunction",
        parent=dev_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This device is intended for detecting, identifying, or screening for low left ventricular ejection fraction "
            "(LVEF) or related cardiac dysfunction."
        ),
        node=lvef_node,
        sources=dev.fda_url,
        additional_instruction=(
            "Look for explicit indications such as 'detect low LVEF', 'screen for decreased ejection fraction', or "
            "phrasing indicating identification of reduced LVEF or cardiac dysfunction tied to LVEF."
        ),
    )

    # 3) Verifiability (critical leaf) – confirm the FDA page corresponds to the device and key registry details
    verif_node = evaluator.add_leaf(
        id=f"device_{device_display_idx}_verifiability",
        desc="The provided device information is verifiable via the referenced FDA database entry",
        parent=dev_node,
        critical=True,
    )
    # Build a robust, tolerant claim that references key fields
    name_str = dev.device_name or ""
    manuf_str = dev.manufacturer or ""
    knum_str = dev.k_number or ""
    cdate_str = dev.clearance_date or ""
    claim = (
        f"This FDA 510(k) page corresponds to the device '{name_str}' by '{manuf_str}', "
        f"with 510(k) number '{knum_str}', and clearance/decision date '{cdate_str}'."
    )
    await evaluator.verify(
        claim=claim,
        node=verif_node,
        sources=dev.fda_url,
        additional_instruction=(
            "Verify that the FDA page lists the device name, manufacturer/applicant, the exact 510(k) number, "
            "and a decision/clearance date matching the provided values. Allow minor formatting differences or synonyms "
            "like 'Applicant'/'Manufacturer' and 'Decision Date'/'Clearance Date'."
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
    Evaluate an answer for the FDA AI-enabled ECG LVEF devices in Q3 2025 task.
    """

    # Initialize evaluator (root is parallel aggregation for two independent devices)
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

    # Extract all devices mentioned in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_devices(),
        template_class=DevicesExtraction,
        extraction_name="extracted_devices",
    )

    # Keep only the first two devices; pad with empty if fewer provided
    devices: List[DeviceInfo] = list(extracted.devices[:2])
    while len(devices) < 2:
        devices.append(DeviceInfo())

    # Optional: record timeframe info
    evaluator.add_custom_info(
        info={
            "timeframe": "Q3 2025",
            "start_date": str(Q3_2025_START),
            "end_date": str(Q3_2025_END),
            "devices_extracted_count": len(extracted.devices),
        },
        info_type="context",
        info_name="evaluation_context",
    )

    # Build verification trees for the two devices
    for idx, dev in enumerate(devices):
        await verify_device(evaluator, root, dev, idx)

    # Return summary
    return evaluator.get_summary()