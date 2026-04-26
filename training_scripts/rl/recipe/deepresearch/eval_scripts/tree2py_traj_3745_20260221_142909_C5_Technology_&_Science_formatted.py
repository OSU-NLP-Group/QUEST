import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "smartphones_satellite_ces2026"
TASK_DESCRIPTION = """
I'm researching smartphones with satellite emergency communication capabilities for use in remote areas of the United States where cellular coverage is unavailable. I want to focus on devices from manufacturers that showcased their products at CES 2026 (which took place January 6-9, 2026 in Las Vegas).

Please identify four distinct smartphone models that meet all of the following criteria:

1. The device must support satellite-based emergency communication (such as Emergency SOS via satellite or Satellite SOS) that functions without cellular or Wi-Fi coverage
2. The manufacturer must have been an exhibitor at CES 2026
3. The satellite emergency feature must be available for use in the United States
4. The device must be commercially available as of February 2026

For each smartphone model, provide:
- The specific model name
- The manufacturer
- A brief description of the satellite emergency feature
- Verification that the manufacturer exhibited at CES 2026 with a reference URL
- Verification of the satellite emergency feature with a reference URL
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DeviceInfo(BaseModel):
    """Information for a single smartphone device entry."""
    model_name: Optional[str] = None
    manufacturer: Optional[str] = None
    satellite_feature_desc: Optional[str] = None

    # Evidence URLs explicitly cited in the answer
    ces_exhibitor_urls: List[str] = Field(default_factory=list)           # URLs that verify the manufacturer exhibited at CES 2026
    satellite_feature_urls: List[str] = Field(default_factory=list)       # URLs that verify satellite emergency capability and no-cell/Wi-Fi operation

    # Additional indications required by rubric (presence checks in answer)
    us_availability_note: Optional[str] = None                            # Indication that the feature is available in the US
    commercially_available_note: Optional[str] = None                     # Indication device was commercially available by Feb 2026
    clear_sky_requirement_note: Optional[str] = None                      # Indication satellite connectivity needs outdoor clear sky

    # Optional product/ecommerce URLs (if the answer provides them)
    availability_urls: List[str] = Field(default_factory=list)


class DevicesExtraction(BaseModel):
    """Top-level extraction structure containing up to four devices."""
    devices: List[DeviceInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_devices() -> str:
    return """
    Extract up to four distinct smartphone entries exactly as provided in the answer, capturing the following fields for each:

    REQUIRED FIELDS:
    - model_name: The specific smartphone model name
    - manufacturer: The manufacturer name
    - satellite_feature_desc: A brief description of the satellite-based emergency communication feature

    REQUIRED CITATION FIELDS:
    - ces_exhibitor_urls: A list of URLs cited in the answer that verify the manufacturer was an exhibitor at CES 2026 (Jan 6-9, 2026, Las Vegas). These may include official CES exhibitor directory pages, CES newsroom pages, booth listings, or official manufacturer press/news specifically stating exhibiting at CES 2026.
    - satellite_feature_urls: A list of URLs cited in the answer that verify the device supports satellite-based emergency communication which works without cellular or Wi‑Fi coverage (e.g., “Emergency SOS via satellite”, “Satellite SOS”).

    ADDITIONAL INDICATIONS (presence only, not necessarily requiring URLs):
    - us_availability_note: If the answer states the satellite emergency feature is available for use in the United States, extract the exact phrase; otherwise return null.
    - commercially_available_note: If the answer states the device was commercially available as of February 2026, extract the exact phrase; otherwise return null.
    - clear_sky_requirement_note: If the answer states the satellite connectivity requires being outdoors with a clear view of the sky (or equivalent wording like unobstructed view), extract that phrase; otherwise return null.

    OPTIONAL:
    - availability_urls: Any product page, official store page, carrier page, or other URLs cited that relate to commercial availability; if none are cited, return an empty list.

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer (plain URLs or markdown links).
    - Do not invent or infer URLs.
    - Normalize missing fields to null and missing URL lists to empty arrays.
    - If the answer contains more than four entries, only extract the first four.
    - If the answer contains fewer than four entries, extract what is present (others will be handled by the evaluation script).
    - Preserve exact text snippets for the *_note fields.

    Return JSON with a single key 'devices' which is an array of device objects following the schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len([u for u in urls if _is_nonempty_str(u)]) > 0)


def _normalize_name(s: Optional[str]) -> Optional[str]:
    return s.strip().lower() if _is_nonempty_str(s) else None


# --------------------------------------------------------------------------- #
# Verification subroutine for one device                                      #
# --------------------------------------------------------------------------- #
async def verify_one_device(
    evaluator: Evaluator,
    root: Any,
    device: DeviceInfo,
    idx_one_based: int
) -> None:
    """
    Build the verification subtree and run checks for a single device.
    """
    dev_node = evaluator.add_parallel(
        id=f"device_{idx_one_based}",
        desc=f"{idx_one_based}st smartphone model (meets all criteria; includes required fields and citations)" if idx_one_based == 1 else (
             f"{idx_one_based}nd smartphone model (meets all criteria; includes required fields and citations)" if idx_one_based == 2 else (
             f"{idx_one_based}rd smartphone model (meets all criteria; includes required fields and citations)" if idx_one_based == 3 else
             f"{idx_one_based}th smartphone model (meets all criteria; includes required fields and citations)")),
        parent=root,
        critical=False
    )

    # ---- Presence checks (critical) ----
    evaluator.add_custom_node(
        result=_is_nonempty_str(device.model_name),
        id=f"d{idx_one_based}_model_name_provided",
        desc="Provides the specific smartphone model name",
        parent=dev_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(device.manufacturer),
        id=f"d{idx_one_based}_manufacturer_provided",
        desc="Provides the manufacturer name",
        parent=dev_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(device.satellite_feature_desc),
        id=f"d{idx_one_based}_satellite_feature_description_provided",
        desc="Provides a brief description of the satellite emergency feature",
        parent=dev_node,
        critical=True
    )

    # ---- Evidence gating nodes (critical) to avoid source-less verification ----
    ces_sources_node = evaluator.add_custom_node(
        result=_has_urls(device.ces_exhibitor_urls),
        id=f"d{idx_one_based}_ces_sources_provided",
        desc="CES exhibitor verification source URL(s) provided",
        parent=dev_node,
        critical=True
    )
    sat_sources_node = evaluator.add_custom_node(
        result=_has_urls(device.satellite_feature_urls),
        id=f"d{idx_one_based}_satellite_sources_provided",
        desc="Satellite feature verification source URL(s) provided",
        parent=dev_node,
        critical=True
    )

    # ---- CES exhibitor verification (critical) ----
    ces_leaf = evaluator.add_leaf(
        id=f"d{idx_one_based}_ces_exhibitor_verified",
        desc="Provides a reference URL verifying the manufacturer exhibited at CES 2026",
        parent=dev_node,
        critical=True
    )
    manufacturer = device.manufacturer or ""
    ces_claim = f"The manufacturer '{manufacturer}' was an exhibitor at CES 2026 (January 6–9, 2026, Las Vegas)."
    await evaluator.verify(
        claim=ces_claim,
        node=ces_leaf,
        sources=device.ces_exhibitor_urls,
        additional_instruction=(
            "Verify that the cited page(s) explicitly indicate the company exhibited at CES 2026. "
            "Accept CES exhibitor directory entries, official CES pages, booth listings, or official "
            "company press/news clearly stating presence at CES 2026. The evidence must refer to the 2026 event."
        ),
        extra_prerequisites=[ces_sources_node]  # Explicit gating dependency
    )

    # ---- Satellite feature verification (critical) ----
    sat_leaf = evaluator.add_leaf(
        id=f"d{idx_one_based}_satellite_no_cell_wifi_verified",
        desc="Provides a reference URL verifying the device supports satellite-based emergency communication that functions without cellular or Wi‑Fi coverage",
        parent=dev_node,
        critical=True
    )
    model_name = device.model_name or ""
    sat_claim = (
        f"The smartphone '{model_name}' supports satellite-based emergency communication that operates "
        f"without cellular or Wi‑Fi coverage (e.g., Emergency SOS via satellite or Satellite SOS)."
    )
    await evaluator.verify(
        claim=sat_claim,
        node=sat_leaf,
        sources=device.satellite_feature_urls,
        additional_instruction=(
            "Check the cited page(s) for explicit statements that satellite emergency messaging works when there is "
            "no cellular service and no Wi‑Fi. Accept feature names like 'Emergency SOS via satellite', 'Satellite SOS', "
            "or equivalent phrasing indicating off-grid emergency messaging via satellite."
        ),
        extra_prerequisites=[sat_sources_node]  # Explicit gating dependency
    )

    # ---- Presence checks for remaining rubric items (critical) ----
    evaluator.add_custom_node(
        result=_is_nonempty_str(device.us_availability_note),
        id=f"d{idx_one_based}_feature_available_in_us",
        desc="Indicates the satellite emergency feature is available for use in the United States",
        parent=dev_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(device.commercially_available_note),
        id=f"d{idx_one_based}_commercially_available_feb_2026",
        desc="Indicates the device is commercially available as of February 2026",
        parent=dev_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(device.clear_sky_requirement_note),
        id=f"d{idx_one_based}_requires_outdoor_clear_sky",
        desc="Indicates the satellite connectivity requires an outdoor location with a clear view of the sky",
        parent=dev_node,
        critical=True
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
    Evaluate an answer for the CES 2026 satellite emergency smartphones task.
    """
    # Initialize evaluator and root
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
        default_model=model
    )

    # Extract structured devices from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_devices(),
        template_class=DevicesExtraction,
        extraction_name="devices_extraction"
    )

    # Keep only the first four devices; pad with empty entries if fewer
    devices: List[DeviceInfo] = list(extracted.devices[:4])
    while len(devices) < 4:
        devices.append(DeviceInfo())

    # Critical: all four model names must be distinct
    norm_names = [_normalize_name(d.model_name) for d in devices]
    all_present = all(n is not None for n in norm_names)
    all_distinct = len(set(n for n in norm_names if n is not None)) == 4 if all_present else False
    evaluator.add_custom_node(
        result=(all_present and all_distinct),
        id="distinct_models",
        desc="All four provided smartphone model names are distinct (no duplicates)",
        parent=root,
        critical=True
    )

    # Build and verify subtrees for each device
    for i in range(1, 5):
        await verify_one_device(evaluator, root, devices[i - 1], i)

    # Optional: record basic counts
    evaluator.add_custom_info(
        info={
            "extracted_device_count": len(extracted.devices),
            "evaluated_device_count": 4,
        },
        info_type="summary_stats"
    )

    # Return evaluation summary
    return evaluator.get_summary()