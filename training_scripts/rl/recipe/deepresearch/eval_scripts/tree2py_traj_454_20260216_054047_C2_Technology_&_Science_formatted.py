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
TASK_ID = "airpods_pro_hearing_aid_feature"
TASK_DESCRIPTION = (
    "I'm interested in using Apple's new hearing aid feature on AirPods Pro to help with my mild hearing loss. "
    "Before purchasing, I need to understand the complete technical and regulatory details. Please provide comprehensive "
    "information about this feature, including: (1) Which specific AirPods Pro model(s) are compatible with the hearing "
    "aid feature? (2) What is the minimum iOS or iPadOS version required on my iPhone or iPad? (3) When did the FDA authorize "
    "this hearing aid feature? (4) When did this feature officially launch for users in the United States? "
    "(5) As of February 2026, is this hearing aid feature currently available in both the United States and Canada? "
    "For each piece of information, please include supporting reference URLs from official or reliable sources."
)

# Optional reference info (for logging/ground truth context only)
EXPECTED_INFO = {
    "min_ios_ipados_version": "iOS 18.1 or iPadOS 18.1 or later",
    "min_firmware_version": "7B19 or later",
    "fda_authorization_date": "September 12, 2024",
    "us_launch_date": "October 28, 2024",
    "availability_reference_month": "February 2026"
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HardwareExtraction(BaseModel):
    models: List[str] = Field(default_factory=list, description="List of AirPods Pro model names mentioned as compatible")
    firmware_version: Optional[str] = None
    min_os_version: Optional[str] = None
    hardware_urls: List[str] = Field(default_factory=list, description="URLs cited for hardware/OS/firmware compatibility")


class RegulatoryExtraction(BaseModel):
    fda_authorization_date: Optional[str] = None
    us_launch_date: Optional[str] = None
    regulatory_urls: List[str] = Field(default_factory=list, description="URLs cited for FDA authorization and US launch dates")


class GeographicExtraction(BaseModel):
    us_availability: Optional[str] = None  # e.g., "available", "yes", "not available", "no"
    canada_availability: Optional[str] = None
    canada_notes: Optional[str] = None  # e.g., mentions Health Canada approval and/or provincial restrictions
    geographic_urls: List[str] = Field(default_factory=list, description="URLs cited for geographic availability (US/Canada)")


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hardware() -> str:
    return """
Extract the hardware/software requirements for Apple's AirPods Pro hearing aid feature as explicitly stated in the answer.

Return a JSON object with:
- models: array of the specific AirPods Pro model(s) named as compatible (e.g., "AirPods Pro (2nd generation)", "AirPods Pro 2")
- firmware_version: the minimum AirPods firmware/build version required for the feature (e.g., "7B19" or "7B19 or later")
- min_os_version: the minimum iOS or iPadOS version required (e.g., "iOS 18.1", "iOS 18.1 or later", "iPadOS 18.1 or later")
- hardware_urls: array of all URLs the answer provides to support the hardware/OS/firmware requirements

Rules:
- Extract exactly what appears in the answer text. Do not infer any values.
- If an item is missing, set it to null (for strings) or [] (for arrays).
- Include only URLs explicitly present in the answer.
"""


def prompt_extract_regulatory() -> str:
    return """
Extract the regulatory timeline details for Apple's AirPods Pro hearing aid feature as explicitly stated in the answer.

Return a JSON object with:
- fda_authorization_date: the date the U.S. FDA authorized the feature (e.g., "September 12, 2024")
- us_launch_date: the date the feature launched for users in the United States (e.g., "October 28, 2024")
- regulatory_urls: array of all URLs the answer provides to support these dates

Rules:
- Extract dates as strings exactly as written in the answer.
- If an item is missing, set it to null (for strings) or [] (for arrays).
- Include only URLs explicitly present in the answer.
"""


def prompt_extract_geographic() -> str:
    return """
Extract the geographic availability status for Apple's AirPods Pro hearing aid feature as explicitly stated in the answer.

Return a JSON object with:
- us_availability: the stated availability in the United States as of February 2026 (e.g., "available", "yes", "not available", "no")
- canada_availability: the stated availability in Canada as of February 2026 (e.g., "not available", "yes", "no")
- canada_notes: any extra notes provided about Canada (e.g., "Health Canada approval in December 2024", "provincial regulatory restrictions")
- geographic_urls: array of all URLs the answer provides to support availability in the U.S. and/or Canada

Rules:
- Extract exactly what appears in the answer text. Do not infer or add.
- If an item is missing, set it to null (for strings) or [] (for arrays).
- Include only URLs explicitly present in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _list_to_english(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _has_valid_urls(urls: List[str]) -> bool:
    return any(isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")) for u in urls)


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_hardware_checks(
    evaluator: Evaluator,
    parent_node,
    hardware: HardwareExtraction
) -> None:
    """
    Build and execute the 'Compatible_Hardware_Requirements' subtree:
      – AirPods model identification (critical)
      – Firmware version requirement (critical)
      – Minimum iOS/iPadOS version (critical)
      – Reference URL presence (non-critical)
    """
    hw_node = evaluator.add_sequential(
        id="Compatible_Hardware_Requirements",
        desc="Compatible AirPods model, firmware version, and minimum iOS/iPadOS version required for the hearing aid feature",
        parent=parent_node,
        critical=False
    )

    # 1) AirPods Model Identification (Critical)
    models_text = _list_to_english(hardware.models)
    model_leaf = evaluator.add_leaf(
        id="AirPods_Model_Identification",
        desc="Correctly identifies which AirPods Pro generation(s) support the hearing aid feature (AirPods Pro 2 and/or AirPods Pro 3, NOT AirPods Pro 1st generation)",
        parent=hw_node,
        critical=True
    )
    claim_models = f"The AirPods Pro hearing aid feature is compatible with the following AirPods Pro model(s): {models_text}."
    await evaluator.verify(
        claim=claim_models,
        node=model_leaf,
        sources=hardware.hardware_urls,
        additional_instruction=(
            "Verify that the cited page(s) explicitly state which AirPods Pro generations support the 'Hearing Aid' feature. "
            "Treat naming variants like 'AirPods Pro (2nd generation)' and 'AirPods Pro 2' as equivalent. "
            "If the page(s) contradict the claim or do not support it, mark as not supported."
        )
    )

    # 2) Firmware Version (Critical)
    fw_leaf = evaluator.add_leaf(
        id="Firmware_Version",
        desc="Specifies the minimum firmware version required for AirPods Pro 2 (firmware version 7B19 or later)",
        parent=hw_node,
        critical=True
    )
    fw_value = hardware.firmware_version or ""
    claim_fw = f"The minimum AirPods firmware version required for the hearing aid feature is {fw_value}."
    await evaluator.verify(
        claim=claim_fw,
        node=fw_leaf,
        sources=hardware.hardware_urls,
        additional_instruction=(
            "The page should indicate a specific firmware build requirement (e.g., '7B19' or '7B19 or later') for AirPods Pro (2nd generation). "
            "Minor phrasing differences are acceptable; focus on whether the stated minimum firmware is supported."
        )
    )

    # 3) Minimum iOS/iPadOS Version (Critical)
    os_leaf = evaluator.add_leaf(
        id="Minimum_iOS_iPadOS_Version",
        desc="Specifies the minimum iOS or iPadOS version required (iOS 18.1 or iPadOS 18.1 or later)",
        parent=hw_node,
        critical=True
    )
    min_os = hardware.min_os_version or ""
    claim_os = f"The minimum iOS or iPadOS version required for the hearing aid feature is {min_os}."
    await evaluator.verify(
        claim=claim_os,
        node=os_leaf,
        sources=hardware.hardware_urls,
        additional_instruction=(
            "Accept equivalent phrasing such as 'iOS 18.1 or later' or 'iPadOS 18.1 or later'. "
            "If both iOS and iPadOS are mentioned separately, ensure the minimum version matches what the claim states."
        )
    )

    # 4) Reference URL presence (Non-Critical)
    evaluator.add_custom_node(
        result=_has_valid_urls(hardware.hardware_urls),
        id="Reference_URL_Hardware",
        desc="Provides a valid reference URL supporting the hardware compatibility information",
        parent=hw_node,
        critical=False
    )


async def build_regulatory_checks(
    evaluator: Evaluator,
    parent_node,
    regulatory: RegulatoryExtraction
) -> None:
    """
    Build and execute the 'FDA_Regulatory_Status' subtree:
      – FDA Authorization Date (critical)
      – US Launch Date (critical)
      – Reference URL presence (non-critical)
    """
    reg_node = evaluator.add_sequential(
        id="FDA_Regulatory_Status",
        desc="FDA authorization date and US launch date for the hearing aid feature",
        parent=parent_node,
        critical=False
    )

    # 1) FDA Authorization Date (Critical)
    fda_leaf = evaluator.add_leaf(
        id="FDA_Authorization_Date",
        desc="Provides the date when the FDA authorized the hearing aid feature (September 12, 2024)",
        parent=reg_node,
        critical=True
    )
    fda_date = regulatory.fda_authorization_date or ""
    claim_fda = f"The U.S. FDA authorized the AirPods Pro hearing aid feature on {fda_date}."
    await evaluator.verify(
        claim=claim_fda,
        node=fda_leaf,
        sources=regulatory.regulatory_urls,
        additional_instruction=(
            "Look for official or reliable sources indicating FDA authorization (e.g., De Novo classification/marketing authorization). "
            "The page should clearly indicate the authorization date matching the claim (allowing minor formatting differences)."
        )
    )

    # 2) US Launch Date (Critical)
    us_launch_leaf = evaluator.add_leaf(
        id="US_Launch_Date",
        desc="Provides the date when the feature became available to users in the United States (October 28, 2024)",
        parent=reg_node,
        critical=True
    )
    us_launch_date = regulatory.us_launch_date or ""
    claim_launch = f"The AirPods Pro hearing aid feature launched for users in the United States on {us_launch_date}."
    await evaluator.verify(
        claim=claim_launch,
        node=us_launch_leaf,
        sources=regulatory.regulatory_urls,
        additional_instruction=(
            "Accept phrasing like 'available', 'rollout', or 'launched'. "
            "The page(s) should clearly indicate the first availability date for U.S. users matching the claim."
        )
    )

    # 3) Reference URL presence (Non-Critical)
    evaluator.add_custom_node(
        result=_has_valid_urls(regulatory.regulatory_urls),
        id="Reference_URL_Regulatory",
        desc="Provides a valid reference URL supporting the regulatory dates",
        parent=reg_node,
        critical=False
    )


async def build_geographic_checks(
    evaluator: Evaluator,
    parent_node,
    geographic: GeographicExtraction
) -> None:
    """
    Build and execute the 'Geographic_Availability' subtree:
      – United States availability (critical)
      – Canada availability status (critical)
      – Reference URL presence (non-critical)
    """
    geo_node = evaluator.add_parallel(
        id="Geographic_Availability",
        desc="Current availability status in the United States and Canada as of February 2026",
        parent=parent_node,
        critical=False
    )

    # 1) United States Availability (Critical)
    us_leaf = evaluator.add_leaf(
        id="United_States_Availability",
        desc="Confirms that the hearing aid feature is available in the United States",
        parent=geo_node,
        critical=True
    )
    # Build a clear, time-scoped claim
    claim_us = (
        "As of February 2026, the AirPods Pro hearing aid feature is available to users in the United States."
    )
    await evaluator.verify(
        claim=claim_us,
        node=us_leaf,
        sources=geographic.geographic_urls,
        additional_instruction=(
            "Verify that the page(s) confirm availability in the United States. "
            "Accept phrasing variants like 'available to U.S. users' or 'released in the U.S.'."
        )
    )

    # 2) Canada Availability Status (Critical)
    ca_leaf = evaluator.add_leaf(
        id="Canada_Availability_Status",
        desc="Confirms that the hearing aid feature is NOT available in Canada as of February 2026, despite Health Canada approval in December 2024, due to provincial regulatory restrictions",
        parent=geo_node,
        critical=True
    )
    claim_ca = (
        "As of February 2026, the AirPods Pro hearing aid feature is not available in Canada, "
        "despite Health Canada approval in December 2024, primarily due to provincial regulatory restrictions."
    )
    await evaluator.verify(
        claim=claim_ca,
        node=ca_leaf,
        sources=geographic.geographic_urls,
        additional_instruction=(
            "Look for statements that the feature is not yet available in Canada as of February 2026. "
            "Also check for mentions that Health Canada granted approval (e.g., in December 2024) but provincial regulations "
            "or restrictions prevent availability. If evidence contradicts any part of the claim, mark as not supported."
        )
    )

    # 3) Reference URL presence (Non-Critical)
    evaluator.add_custom_node(
        result=_has_valid_urls(geographic.geographic_urls),
        id="Reference_URL_Geographic",
        desc="Provides a valid reference URL supporting the geographic availability information",
        parent=geo_node,
        critical=False
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
    Evaluate an answer for the AirPods Pro hearing aid feature information task.
    Builds a verification tree that checks hardware compatibility, regulatory milestones, and geographic availability,
    grounding each factual claim against the URLs cited in the answer where applicable.
    """
    # Initialize evaluator (root is non-critical parallel by default)
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

    # Extract information concurrently
    hardware_task = evaluator.extract(
        prompt=prompt_extract_hardware(),
        template_class=HardwareExtraction,
        extraction_name="hardware_requirements"
    )
    regulatory_task = evaluator.extract(
        prompt=prompt_extract_regulatory(),
        template_class=RegulatoryExtraction,
        extraction_name="regulatory_timeline"
    )
    geographic_task = evaluator.extract(
        prompt=prompt_extract_geographic(),
        template_class=GeographicExtraction,
        extraction_name="geographic_availability"
    )

    hardware, regulatory, geographic = await asyncio.gather(hardware_task, regulatory_task, geographic_task)

    # Optionally record expected reference info for transparency
    evaluator.add_ground_truth(
        gt_info=EXPECTED_INFO,
        gt_type="expected_reference_info"
    )

    # Top-level node (keep non-critical to allow partial scoring across sections)
    main_node = evaluator.add_parallel(
        id="AirPods_Pro_Hearing_Aid_Feature_Information",
        desc="Comprehensive information about Apple's AirPods Pro hearing aid feature, including technical requirements, regulatory approval, and availability",
        parent=root,
        critical=False
    )

    # Build and run subtrees
    await build_hardware_checks(evaluator, main_node, hardware)
    await build_regulatory_checks(evaluator, main_node, regulatory)
    await build_geographic_checks(evaluator, main_node, geographic)

    # Return evaluation summary
    return evaluator.get_summary()