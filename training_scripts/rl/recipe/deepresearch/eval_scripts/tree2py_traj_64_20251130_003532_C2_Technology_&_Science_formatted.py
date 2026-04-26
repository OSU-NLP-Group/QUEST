import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "apple_wearable_s9_2000nits_2023"
TASK_DESCRIPTION = """
What is the name of the Apple wearable device that was announced on September 12, 2023, uses the S9 SiP processor, has a maximum display brightness of 2000 nits, and became available for purchase on September 22, 2023? Provide the device name, confirm the processor model, state the maximum display brightness, verify the announcement and availability dates, and include reference URLs to support your answer.
"""


# ------------------------------ Data Models ------------------------------ #
class AppleWearableExtraction(BaseModel):
    device_name: Optional[str] = None
    device_category: Optional[str] = None  # e.g., "Apple Watch", "watch"
    processor_model: Optional[str] = None  # e.g., "S9 SiP"
    neural_engine_details: Optional[str] = None  # e.g., "4-core Neural Engine"
    max_brightness: Optional[str] = None  # e.g., "2000 nits"
    announcement_date: Optional[str] = None  # e.g., "September 12, 2023"
    availability_date: Optional[str] = None  # e.g., "September 22, 2023"
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------- Extraction Prompts --------------------------- #
def prompt_extract_device_info() -> str:
    return """
    Extract the Apple wearable device information explicitly stated in the answer.

    Return a JSON object with the following fields:
    - device_name: The single, specific device name identified in the answer (e.g., "Apple Watch Series 9"). If multiple names are mentioned, choose the one the answer ties to the requested specs and dates. If unclear, return the first specific device name mentioned. If none, return null.
    - device_category: The device category if stated (e.g., "Apple Watch"). If not present, return null.
    - processor_model: The processor model name if stated (e.g., "S9 SiP"). If not present, return null.
    - neural_engine_details: Any explicit mention about the Neural Engine (e.g., "4-core Neural Engine"). If not present, return null.
    - max_brightness: The maximum display brightness value if stated (e.g., "2000 nits", "peak 2000 nits"). If not present, return null.
    - announcement_date: The announcement date if stated (e.g., "September 12, 2023"). If not present, return null.
    - availability_date: The availability date if stated (e.g., "September 22, 2023"). If not present, return null.
    - reference_urls: All reference URLs included in the answer. Extract actual URLs only (including those inside markdown links). If none, return an empty list.

    Only extract information that is explicitly present in the answer text. Do not infer or invent any values.
    """


# ------------------------- Helper / Verification Logic ------------------------- #
def _safe_name(extracted: AppleWearableExtraction) -> str:
    return extracted.device_name or "the device"


async def build_references_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: AppleWearableExtraction,
) -> None:
    refs_node = evaluator.add_parallel(
        id="References",
        desc="Check the answer includes supporting reference URL(s).",
        parent=parent_node,
        critical=True,
    )

    # Single leaf: check that reference URLs are present
    evaluator.add_custom_node(
        result=(len(extracted.reference_urls) > 0),
        id="Reference_URLs_Support_Claims",
        desc="The answer includes reference URL(s) that support the device identity and the stated specs/dates.",
        parent=refs_node,
        critical=True,
    )


async def build_device_identity_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: AppleWearableExtraction,
) -> None:
    identity_node = evaluator.add_parallel(
        id="Device_Identity",
        desc="Check the answer identifies the wearable device in the Apple Watch category.",
        parent=parent_node,
        critical=True,
    )

    # Existence of a single specific device name
    evaluator.add_custom_node(
        result=(extracted.device_name is not None and extracted.device_name.strip() != ""),
        id="Device_Name_Provided",
        desc="The answer provides a single specific device name.",
        parent=identity_node,
        critical=True,
    )

    # Confirm device is Apple Watch category (verify via URLs)
    cat_leaf = evaluator.add_leaf(
        id="Device_Category_Apple_Watch",
        desc="The answer confirms the device is a wearable product in the Apple Watch category.",
        parent=identity_node,
        critical=True,
    )
    claim = f"The device named '{_safe_name(extracted)}' is a wearable in the Apple Watch product category."
    await evaluator.verify(
        claim=claim,
        node=cat_leaf,
        sources=extracted.reference_urls,
        additional_instruction="Use the provided URLs to confirm that the device belongs to the Apple Watch product line. Minor naming variations are acceptable."
    )


async def build_processor_spec_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: AppleWearableExtraction,
) -> None:
    proc_node = evaluator.add_parallel(
        id="Processor_Specifications",
        desc="Check the processor requirements stated in the constraints.",
        parent=parent_node,
        critical=True,
    )

    # Processor model: S9 SiP
    s9_leaf = evaluator.add_leaf(
        id="Processor_Model_S9_SiP",
        desc="The answer states the device uses the S9 SiP processor.",
        parent=proc_node,
        critical=True,
    )
    s9_claim = f"The device '{_safe_name(extracted)}' uses Apple's S9 SiP processor."
    await evaluator.verify(
        claim=s9_claim,
        node=s9_leaf,
        sources=extracted.reference_urls,
        additional_instruction="Confirm via the provided URLs that the device is equipped with the S9 SiP processor. Accept minor phrasing variations like 'S9 chip' or 'S9 SiP'."
    )

    # Neural Engine: 4-core
    ne_leaf = evaluator.add_leaf(
        id="Neural_Engine_4_Core",
        desc="The answer states/verifies the processor includes a 4-core Neural Engine.",
        parent=proc_node,
        critical=True,
    )
    ne_claim = f"The S9 SiP used in '{_safe_name(extracted)}' includes a 4-core Neural Engine."
    await evaluator.verify(
        claim=ne_claim,
        node=ne_leaf,
        sources=extracted.reference_urls,
        additional_instruction="Verify from the provided URLs that the S9 SiP includes a 4-core Neural Engine. Allow small wording variations (e.g., '4‑core Neural Engine')."
    )


async def build_display_spec_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: AppleWearableExtraction,
) -> None:
    display_node = evaluator.add_parallel(
        id="Display_Specification",
        desc="Check the maximum display brightness requirement.",
        parent=parent_node,
        critical=True,
    )

    bright_leaf = evaluator.add_leaf(
        id="Maximum_Brightness_2000_Nits",
        desc="The answer states the maximum display brightness is 2000 nits.",
        parent=display_node,
        critical=True,
    )
    bright_claim = f"The maximum display brightness of '{_safe_name(extracted)}' is 2000 nits."
    await evaluator.verify(
        claim=bright_claim,
        node=bright_leaf,
        sources=extracted.reference_urls,
        additional_instruction="Confirm via the URLs that the device reaches 2000 nits peak or maximum brightness. Accept phrasing like 'up to 2000 nits' or 'peak 2000 nits'."
    )


async def build_timeline_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: AppleWearableExtraction,
) -> None:
    time_node = evaluator.add_parallel(
        id="Timeline_Verification",
        desc="Check announcement and availability timing requirements.",
        parent=parent_node,
        critical=True,
    )

    ann_leaf = evaluator.add_leaf(
        id="Announcement_At_Apple_Event_Sep_12_2023",
        desc="The answer states the device was announced at Apple's September 12, 2023 event.",
        parent=time_node,
        critical=True,
    )
    ann_claim = f"The device '{_safe_name(extracted)}' was announced at Apple's event on September 12, 2023."
    await evaluator.verify(
        claim=ann_claim,
        node=ann_leaf,
        sources=extracted.reference_urls,
        additional_instruction="Use the provided URLs (e.g., Apple press release or newsroom) to confirm announcement on Sep 12, 2023."
    )

    avail_leaf = evaluator.add_leaf(
        id="Availability_Date_Sep_22_2023",
        desc="The answer states the device became available for purchase starting September 22, 2023.",
        parent=time_node,
        critical=True,
    )
    avail_claim = f"The device '{_safe_name(extracted)}' became available for purchase starting September 22, 2023."
    await evaluator.verify(
        claim=avail_claim,
        node=avail_leaf,
        sources=extracted.reference_urls,
        additional_instruction="Confirm via the URLs that retail availability began on Sep 22, 2023 (phrasing like 'available starting September 22, 2023' is acceptable)."
    )


# -------------------------- Main Evaluation Entry -------------------------- #
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_device_info(),
        template_class=AppleWearableExtraction,
        extraction_name="apple_wearable_extraction",
    )

    # Build top-level critical task node
    task_node = evaluator.add_parallel(
        id="Device_Identification_Task",
        desc="Identify the Apple wearable device meeting the given constraints and report the requested attributes with supporting URLs.",
        parent=root,
        critical=True,
    )

    # References check first (critical gating)
    await build_references_checks(evaluator, task_node, extracted)

    # Subtrees
    await build_device_identity_checks(evaluator, task_node, extracted)
    await build_processor_spec_checks(evaluator, task_node, extracted)
    await build_display_spec_checks(evaluator, task_node, extracted)
    await build_timeline_checks(evaluator, task_node, extracted)

    return evaluator.get_summary()