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
TASK_ID = "android16_devices_spec_compare_2026"
TASK_DESCRIPTION = (
    "I'm considering purchasing two specific Android 16 devices released in early 2026 for different purposes: "
    "the Samsung Galaxy S26 Ultra as a primary flagship smartphone and the Clicks Communicator as a focused communication device. "
    "To help me make an informed decision, I need a direct comparison of their key technical specifications. "
    "Please provide the following information for both devices: display size (in inches), battery capacity (in mAh), "
    "and Qi2 wireless charging support (yes/no). Provide these specifications for both the Samsung Galaxy S26 Ultra "
    "and the Clicks Communicator."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class DeviceSpec(BaseModel):
    display_size: Optional[str] = None
    display_size_sources: List[str] = Field(default_factory=list)

    battery_capacity: Optional[str] = None
    battery_sources: List[str] = Field(default_factory=list)

    qi2_support: Optional[str] = None  # e.g., "yes", "no", or any phrase from the answer
    qi2_sources: List[str] = Field(default_factory=list)


class SpecsExtraction(BaseModel):
    samsung_galaxy_s26_ultra: Optional[DeviceSpec] = None
    clicks_communicator: Optional[DeviceSpec] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_specs() -> str:
    return """
    Extract the requested specifications for the following two devices exactly as stated in the answer text.

    Devices:
    - Samsung Galaxy S26 Ultra
    - Clicks Communicator

    For each device, extract these fields:
    1) display_size: The display/screen size as written (prefer inches, but keep exactly as shown in the answer).
    2) display_size_sources: All URLs explicitly cited in the answer that support the display size (collect every relevant URL).
    3) battery_capacity: The battery capacity as written (keep units/format exactly as shown).
    4) battery_sources: All URLs explicitly cited in the answer that support the battery capacity.
    5) qi2_support: The claimed Qi2 wireless charging support status as written (e.g., "yes", "no", "supports Qi2", "does not support Qi2"; keep the exact phrase).
    6) qi2_sources: All URLs explicitly cited in the answer that support the Qi2 support claim.

    Important rules:
    - Only extract URLs that are explicitly present in the answer (including markdown links); do not invent URLs.
    - If a field is not mentioned, set it to null. For a sources field with no URLs, return an empty array [].
    - Do not normalize or change the values beyond capturing them exactly as provided in the answer.

    Return a JSON object with this structure:
    {
      "samsung_galaxy_s26_ultra": {
        "display_size": ...,
        "display_size_sources": [...],
        "battery_capacity": ...,
        "battery_sources": [...],
        "qi2_support": ...,
        "qi2_sources": [...]
      },
      "clicks_communicator": {
        "display_size": ...,
        "display_size_sources": [...],
        "battery_capacity": ...,
        "battery_sources": [...],
        "qi2_support": ...,
        "qi2_sources": [...]
      }
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _value_provided(val: Optional[str]) -> bool:
    return bool(val is not None and str(val).strip() != "")


def _sources_provided(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


def _parse_yes_no(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    s = value.strip().lower()
    # Handle negatives first to avoid "supports" inside "does not support"
    neg_markers = [
        "does not support", "doesn't support", "not support", "no support",
        "no qi2", "without qi2", "unsupported", "not compatible", "no,",
        "no.", "no ", "not qi2"
    ]
    if any(m in s for m in neg_markers):
        return False
    pos_markers = [
        "supports qi2", "support qi2", "qi2 support", "qi2-compatible",
        "qi2 compliant", "qi2-compliant", "qi2 standard", "qi2 wireless",
        "yes", "supports", "compatible with qi2"
    ]
    if any(m in s for m in pos_markers):
        return True
    # Fallback simple exact checks
    if s in {"yes", "y", "true"}:
        return True
    if s in {"no", "n", "false"}:
        return False
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_spec_group_display(
    evaluator: Evaluator,
    parent,
    device_label: str,
    group_id: str,
    group_desc: str,
    value: Optional[str],
    sources: List[str],
) -> None:
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=_value_provided(value),
        id=f"{group_id}_Provided",
        desc=f"{device_label} display size value is provided",
        parent=group_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_sources_provided(sources),
        id=f"{group_id}_Sources_Provided",
        desc=f"{device_label} display size has at least one source URL",
        parent=group_node,
        critical=True
    )

    supported_node = evaluator.add_leaf(
        id=f"{group_id}_Supported",
        desc=f"{device_label} display size is supported by cited sources",
        parent=group_node,
        critical=True
    )
    claim = f"The {device_label} has a display/screen size of '{value}'."
    await evaluator.verify(
        claim=claim,
        node=supported_node,
        sources=sources,
        additional_instruction=(
            "Verify that the cited page(s) explicitly state a display/screen size equivalent to the quoted value. "
            "Allow minor formatting or rounding differences (e.g., 6.8-inch vs 6.8\"). "
            "Treat 'screen size' and 'display size' as equivalent."
        )
    )


async def _verify_spec_group_battery(
    evaluator: Evaluator,
    parent,
    device_label: str,
    group_id: str,
    group_desc: str,
    value: Optional[str],
    sources: List[str],
) -> None:
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=_value_provided(value),
        id=f"{group_id}_Provided",
        desc=f"{device_label} battery capacity value is provided",
        parent=group_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_sources_provided(sources),
        id=f"{group_id}_Sources_Provided",
        desc=f"{device_label} battery capacity has at least one source URL",
        parent=group_node,
        critical=True
    )

    supported_node = evaluator.add_leaf(
        id=f"{group_id}_Supported",
        desc=f"{device_label} battery capacity is supported by cited sources",
        parent=group_node,
        critical=True
    )
    claim = f"The {device_label} has a battery capacity of '{value}'."
    await evaluator.verify(
        claim=claim,
        node=supported_node,
        sources=sources,
        additional_instruction=(
            "Verify that the cited page(s) explicitly state the battery capacity matching the quoted value. "
            "Allow reasonable synonyms like 'typical capacity', and small rounding where appropriate. "
            "Focus on the main battery capacity number as commonly presented in spec sheets."
        )
    )


async def _verify_spec_group_qi2(
    evaluator: Evaluator,
    parent,
    device_label: str,
    group_id: str,
    group_desc: str,
    value: Optional[str],
    sources: List[str],
) -> None:
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=_value_provided(value),
        id=f"{group_id}_Provided",
        desc=f"{device_label} Qi2 support value is provided",
        parent=group_node,
        critical=True
    )

    parsed = _parse_yes_no(value)
    evaluator.add_custom_node(
        result=(parsed is not None),
        id=f"{group_id}_Value_Valid",
        desc=f"{device_label} Qi2 support value is a clear yes/no claim",
        parent=group_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_sources_provided(sources),
        id=f"{group_id}_Sources_Provided",
        desc=f"{device_label} Qi2 support has at least one source URL",
        parent=group_node,
        critical=True
    )

    supported_node = evaluator.add_leaf(
        id=f"{group_id}_Supported",
        desc=f"{device_label} Qi2 support claim is supported by cited sources",
        parent=group_node,
        critical=True
    )

    if parsed is True:
        claim = f"The {device_label} supports the Qi2 wireless charging standard."
    elif parsed is False:
        claim = f"The {device_label} does not support the Qi2 wireless charging standard."
    else:
        # This will likely be skipped due to failed prerequisites, but provide a fallback claim.
        claim = f"The Qi2 support status for the {device_label} is '{value}', as a definitive yes/no claim."

    await evaluator.verify(
        claim=claim,
        node=supported_node,
        sources=sources,
        additional_instruction=(
            "Verify explicitly whether the device supports the Qi2 wireless charging standard. "
            "If the page clearly states Qi2 support (or clearly denies it), count as supported. "
            "Mentions of generic 'Qi' without 'Qi2' should not be treated as Qi2 unless it explicitly says Qi2."
        )
    )


async def _verify_device_specs(
    evaluator: Evaluator,
    root,
    device_node_id: str,
    device_node_desc: str,
    device_label: str,
    specs: DeviceSpec,
    id_prefix: str
) -> None:
    device_node = evaluator.add_parallel(
        id=device_node_id,
        desc=device_node_desc,
        parent=root,
        critical=False
    )

    # Display size
    await _verify_spec_group_display(
        evaluator=evaluator,
        parent=device_node,
        device_label=device_label,
        group_id=f"{id_prefix}_Display_Size",
        group_desc=f"Provide the display size (in inches) of {device_label}",
        value=specs.display_size,
        sources=specs.display_size_sources
    )

    # Battery capacity
    await _verify_spec_group_battery(
        evaluator=evaluator,
        parent=device_node,
        device_label=device_label,
        group_id=f"{id_prefix}_Battery_Capacity",
        group_desc=f"Provide the battery capacity (in mAh) of {device_label}",
        value=specs.battery_capacity,
        sources=specs.battery_sources
    )

    # Qi2 support
    await _verify_spec_group_qi2(
        evaluator=evaluator,
        parent=device_node,
        device_label=device_label,
        group_id=f"{id_prefix}_Wireless_Charging",
        group_desc=f"Indicate whether {device_label} supports Qi2 wireless charging standard",
        value=specs.qi2_support,
        sources=specs.qi2_sources
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
    Evaluate an answer for the Android 16 devices specifications comparison task.
    """
    # Initialize evaluator (root corresponds to "Device_Specifications_Comparison" in the rubric)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Compare technical specifications of Samsung Galaxy S26 Ultra and Clicks Communicator"
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured specs from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_specs(),
        template_class=SpecsExtraction,
        extraction_name="devices_specs_extraction"
    )

    # Build per-device specs (fallback to empty to ensure nodes get created)
    s26_specs = extraction.samsung_galaxy_s26_ultra or DeviceSpec()
    clicks_specs = extraction.clicks_communicator or DeviceSpec()

    # Verify Samsung Galaxy S26 Ultra
    await _verify_device_specs(
        evaluator=evaluator,
        root=root,
        device_node_id="Samsung_Galaxy_S26_Ultra_Specifications",
        device_node_desc="Provide technical specifications for Samsung Galaxy S26 Ultra",
        device_label="Samsung Galaxy S26 Ultra",
        specs=s26_specs,
        id_prefix="S26_Ultra"
    )

    # Verify Clicks Communicator
    await _verify_device_specs(
        evaluator=evaluator,
        root=root,
        device_node_id="Clicks_Communicator_Specifications",
        device_node_desc="Provide technical specifications for Clicks Communicator",
        device_label="Clicks Communicator",
        specs=clicks_specs,
        id_prefix="Clicks"
    )

    # Return the full evaluation summary
    return evaluator.get_summary()