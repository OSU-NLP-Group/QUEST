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
TASK_ID = "verizon_outage_2026"
TASK_DESCRIPTION = (
    "On January 14, 2026, Verizon Wireless experienced a major nationwide outage that left customers in SOS mode for several hours. "
    "Investigate this incident by answering the following:\n\n"
    "1. What exact time (including timezone) did Verizon announce that the outage was resolved on January 14, 2026?\n\n"
    "2. What specific network architecture component caused this outage? Your answer must identify both the core network type "
    "(distinguishing between 5G Standalone, 5G Non-Standalone, or 4G LTE core) and the triggering activity that caused the failure.\n\n"
    "3. Under FCC regulations, carriers must file a final outage report in the Network Outage Reporting System (NORS) within a specific "
    "timeframe after discovering an outage. What is the exact deadline date by which Verizon must submit its final NORS report for this "
    "January 14, 2026 outage? Show your calculation.\n\n"
    "4. The outage put customers' phones into SOS mode, which relies on Apple's Emergency SOS feature. When was the basic Emergency SOS "
    "feature first introduced to iPhone? Provide the specific iOS version and release date (not the satellite version that came later).\n\n"
    "5. Emergency SOS via satellite was introduced later for iPhone 14. What was the launch date for this satellite feature, and what is "
    "the approximate time gap (in years and months) between the introduction of the basic Emergency SOS feature and the satellite version?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OutageResolutionSection(BaseModel):
    resolution_time_with_timezone: Optional[str] = None  # e.g., "6:34 PM ET" or "6:34 PM EST"
    resolution_date: Optional[str] = None                # e.g., "January 14, 2026"
    resolution_sources: List[str] = Field(default_factory=list)


class TechnicalSection(BaseModel):
    core_network_type: Optional[str] = None              # e.g., "5G Standalone (SA)", "5G Non-Standalone (NSA)", "4G LTE core"
    triggering_activity: Optional[str] = None            # e.g., "software update", "configuration change", "routing table update"
    technical_sources: List[str] = Field(default_factory=list)


class RegulationSection(BaseModel):
    final_report_timeframe_days: Optional[str] = None    # e.g., "30", "30 calendar days"
    regulation_sources: List[str] = Field(default_factory=list)
    discovery_date: Optional[str] = None                 # e.g., "January 14, 2026"
    deadline_date: Optional[str] = None                  # e.g., "February 13, 2026"
    calculation_explanation: Optional[str] = None        # free text explanation included in the answer


class EmergencySection(BaseModel):
    basic_ios_version: Optional[str] = None              # e.g., "iOS 10.2"
    basic_release_date: Optional[str] = None             # e.g., "December 13, 2016"
    basic_sources: List[str] = Field(default_factory=list)
    satellite_launch_date: Optional[str] = None          # e.g., "November 15, 2022"
    satellite_sources: List[str] = Field(default_factory=list)
    gap_years_months: Optional[str] = None               # e.g., "5 years 11 months"
    gap_calculation_explanation: Optional[str] = None    # free text explanation included in the answer


class OutageInvestigationExtraction(BaseModel):
    outage_resolution: OutageResolutionSection = Field(default_factory=OutageResolutionSection)
    technical: TechnicalSection = Field(default_factory=TechnicalSection)
    regulation: RegulationSection = Field(default_factory=RegulationSection)
    emergency: EmergencySection = Field(default_factory=EmergencySection)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_investigation() -> str:
    return """
    Extract the structured information about the Verizon January 14, 2026 outage investigation from the answer text.

    Required fields and formatting:
    1) outage_resolution:
       - resolution_time_with_timezone: The exact time string including timezone (format like "HH:MM AM/PM TZ", e.g., "6:34 PM ET", "6:34 PM EST").
       - resolution_date: The date string (e.g., "January 14, 2026").
       - resolution_sources: All URLs explicitly cited in the answer that confirm the resolution time (Verizon official statement or credible news).

    2) technical:
       - core_network_type: The core network type implicated (choose from strings that appear in the answer; examples include "5G Standalone (SA)", "5G Non-Standalone (NSA)", "4G LTE core").
       - triggering_activity: The specific activity or event that triggered the outage (e.g., "software update", "configuration change", "routing change", "maintenance operation").
       - technical_sources: All URLs explicitly cited that support the core network type and/or triggering activity.

    3) regulation:
       - final_report_timeframe_days: The number of days within which the final NORS report must be filed (extract the number string, e.g., "30" or "30 calendar days").
       - regulation_sources: URLs (FCC or legal sources) that explicitly state this timeframe requirement.
       - discovery_date: The outage discovery date used in the answer's calculation (e.g., "January 14, 2026").
       - deadline_date: The computed deadline date for Verizon's final NORS report for this outage (as stated in the answer).
       - calculation_explanation: The calculation explanation text as presented in the answer (if present).

    4) emergency:
       - basic_ios_version: The iOS version when basic Emergency SOS was first introduced (not satellite).
       - basic_release_date: The exact release date for that iOS version.
       - basic_sources: URLs confirming the iOS version and release date for the basic Emergency SOS introduction.
       - satellite_launch_date: The launch date for Emergency SOS via satellite on iPhone 14.
       - satellite_sources: URLs confirming the satellite feature launch date.
       - gap_years_months: The approximate time gap between the basic Emergency SOS introduction and the satellite feature launch, expressed as "X years Y months" (as provided in the answer).
       - gap_calculation_explanation: The explanation of how the gap was calculated (if present).

    IMPORTANT:
    - Extract only what appears in the answer. If a field is missing, set it to null (or empty array for URLs).
    - For URLs, include only valid, complete URLs explicitly mentioned (plain URLs or markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _sources_or_none(urls: Optional[List[str]]) -> Optional[List[str]]:
    """Return None if URLs are empty to avoid passing empty lists."""
    if not urls:
        return None
    return urls


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_outage_resolution(evaluator: Evaluator, parent_node, data: OutageResolutionSection) -> None:
    # Parent node: parallel, critical
    node = evaluator.add_parallel(
        id="Outage_Resolution_Time",
        desc="Provide the exact time (including timezone) when Verizon announced the outage was resolved on January 14, 2026",
        parent=parent_node,
        critical=True
    )

    # Leaf: Resolution_Time_Content (format and presence)
    time_str = data.resolution_time_with_timezone or ""
    fmt_leaf = evaluator.add_leaf(
        id="Resolution_Time_Content",
        desc="State the precise resolution time with timezone (format: HH:MM AM/PM TIMEZONE)",
        parent=node,
        critical=True
    )
    fmt_claim = f"The string '{time_str}' represents a time in the correct format 'HH:MM AM/PM TIMEZONE' (e.g., 6:34 PM ET) for the Verizon outage resolution announcement."
    await evaluator.verify(
        claim=fmt_claim,
        node=fmt_leaf,
        additional_instruction=(
            "Accept common timezone abbreviations like ET/EST/EDT/PT/PST/PDT/CT/CST/CDT, and also spelled-out forms like 'Eastern Time'. "
            "If the time string is empty or clearly not in the requested format, mark Incorrect."
        ),
    )

    # Leaf: URL_Reference_Resolution (source-backed confirmation)
    url_leaf = evaluator.add_leaf(
        id="URL_Reference_Resolution",
        desc="Provide a valid URL from an official Verizon statement or credible news source confirming the resolution time",
        parent=node,
        critical=True
    )
    url_claim = (
        f"Verizon announced the outage was resolved at {time_str} on {data.resolution_date or 'January 14, 2026'}."
    )
    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=_sources_or_none(data.resolution_sources),
        additional_instruction=(
            "Verify that the provided page(s) explicitly confirm the resolution announcement time and date. "
            "If no URL is provided or the page does not support the claim, mark as Not Supported."
        ),
    )


async def verify_technical_cause(evaluator: Evaluator, parent_node, data: TechnicalSection) -> None:
    # Parent node: sequential, critical
    node = evaluator.add_sequential(
        id="Technical_Cause_Identification",
        desc="Identify the specific network architecture component that caused the January 14, 2026 Verizon outage",
        parent=parent_node,
        critical=True
    )

    # Child: Core_Network_Type (parallel, critical)
    core_node = evaluator.add_parallel(
        id="Core_Network_Type",
        desc="Identify the specific core network type involved in the outage (SA vs NSA vs LTE)",
        parent=node,
        critical=True
    )

    # Leaf: Network_Type_Content (source-backed identification)
    core_type = data.core_network_type or ""
    core_leaf = evaluator.add_leaf(
        id="Network_Type_Content",
        desc="Correctly identify the core network architecture type",
        parent=core_node,
        critical=True
    )
    core_claim = (
        f"The outage involved the '{core_type}' core network architecture (e.g., 5G Standalone/SA, 5G Non-Standalone/NSA, or 4G LTE core)."
    )
    await evaluator.verify(
        claim=core_claim,
        node=core_leaf,
        sources=_sources_or_none(data.technical_sources),
        additional_instruction=(
            "Check whether the source explicitly states the implicated core network type. "
            "Accept equivalent phrasings like '5G SA', 'Standalone 5G core', '5G NSA', 'LTE EPC core'. "
            "If sources don't clearly identify the core type or no URL provided, mark Not Supported."
        ),
    )

    # Leaf: URL_Reference_Technical (ensuring a valid supporting URL exists)
    url_core_leaf = evaluator.add_leaf(
        id="URL_Reference_Technical",
        desc="Provide a valid URL from a technical source or news report explicitly stating the core network type involved",
        parent=core_node,
        critical=True
    )
    url_core_claim = (
        "At least one provided URL explicitly identifies the core network type that was involved in the outage."
    )
    await evaluator.verify(
        claim=url_core_claim,
        node=url_core_leaf,
        sources=_sources_or_none(data.technical_sources),
        additional_instruction=(
            "If no URL is provided, or if the URLs do not explicitly identify the core network type, mark Incorrect."
        ),
    )

    # Leaf: Triggering_Activity (source-backed)
    trigger = data.triggering_activity or ""
    trigger_leaf = evaluator.add_leaf(
        id="Triggering_Activity",
        desc="Identify what activity or event triggered the outage in the core network",
        parent=node,
        critical=True
    )
    trigger_claim = f"The outage was triggered by the following activity/event in the core network: {trigger}."
    await evaluator.verify(
        claim=trigger_claim,
        node=trigger_leaf,
        sources=_sources_or_none(data.technical_sources),
        additional_instruction=(
            "Verify that the page(s) explicitly describe the immediate trigger (e.g., software update, configuration change, maintenance operation). "
            "If the sources do not clearly state the triggering activity or no URL is provided, mark Not Supported."
        ),
    )


async def verify_regulatory_deadline(evaluator: Evaluator, parent_node, data: RegulationSection) -> None:
    # Parent node: sequential, critical
    node = evaluator.add_sequential(
        id="Regulatory_Compliance_Deadline",
        desc="Calculate the FCC NORS final report filing deadline for this outage",
        parent=parent_node,
        critical=True
    )

    # Child: Regulation_Identification (parallel, critical)
    reg_node = evaluator.add_parallel(
        id="Regulation_Identification",
        desc="Identify the timeframe requirement for FCC NORS final report submission after outage discovery",
        parent=node,
        critical=True
    )

    # Leaf: Regulation_Timeframe (source-backed)
    timeframe = data.final_report_timeframe_days or ""
    timeframe_leaf = evaluator.add_leaf(
        id="Regulation_Timeframe",
        desc="State the number of days within which the final report must be filed",
        parent=reg_node,
        critical=True
    )
    timeframe_claim = (
        f"Under FCC NORS rules, carriers must file a final outage report within {timeframe} after discovering the outage."
    )
    await evaluator.verify(
        claim=timeframe_claim,
        node=timeframe_leaf,
        sources=_sources_or_none(data.regulation_sources),
        additional_instruction=(
            "Confirm the exact NORS final report timeframe using FCC or authoritative regulatory sources. "
            "If the timeframe (e.g., '30 days') is not explicitly supported by the provided URL(s) or no URL is provided, mark Not Supported."
        ),
    )

    # Leaf: URL_Reference_Regulation (presence/validity)
    url_reg_leaf = evaluator.add_leaf(
        id="URL_Reference_Regulation",
        desc="Provide a valid URL from the FCC or legal source confirming the final report timeframe requirement",
        parent=reg_node,
        critical=True
    )
    url_reg_claim = "The provided URL(s) are authoritative regulatory sources (e.g., FCC) that confirm the NORS final report timeframe."
    await evaluator.verify(
        claim=url_reg_claim,
        node=url_reg_leaf,
        sources=_sources_or_none(data.regulation_sources),
        additional_instruction=(
            "If no URL is provided or the URLs are not authoritative (FCC/regulatory), mark Incorrect."
        ),
    )

    # Leaf: Deadline_Calculation (logical check)
    deadline = data.deadline_date or ""
    disc_date = data.discovery_date or "January 14, 2026"
    calc_leaf = evaluator.add_leaf(
        id="Deadline_Calculation",
        desc="Calculate and provide the exact deadline date by adding the required timeframe to the outage discovery date, showing the calculation",
        parent=node,
        critical=True
    )
    calc_claim = (
        f"Given the outage discovery date is {disc_date} and the NORS final report timeframe is {timeframe}, "
        f"the correct deadline date is {deadline}."
    )
    await evaluator.verify(
        claim=calc_claim,
        node=calc_leaf,
        additional_instruction=(
            "Verify the date arithmetic. Treat '30 calendar days' as adding 30 days to the discovery date. "
            "If the timeframe indicates business days, the answer should reflect such. "
            "Judge correctness solely on whether the provided deadline date matches proper addition from the discovery date."
        ),
    )


async def verify_emergency_feature(evaluator: Evaluator, parent_node, data: EmergencySection) -> None:
    # Parent node: sequential, critical
    node = evaluator.add_sequential(
        id="Related_Emergency_Feature",
        desc="Identify when Apple first introduced the basic Emergency SOS feature to iPhone and track evolution to satellite",
        parent=parent_node,
        critical=True
    )

    # Child: Feature_Introduction_Date (parallel, critical)
    feat_node = evaluator.add_parallel(
        id="Feature_Introduction_Date",
        desc="Identify when the basic Emergency SOS feature was first introduced to iPhone, including iOS version and release date",
        parent=node,
        critical=True
    )

    # Leaf: iOS_Version_and_Date (source-backed)
    ios_ver = data.basic_ios_version or ""
    basic_date = data.basic_release_date or ""
    ios_leaf = evaluator.add_leaf(
        id="iOS_Version_and_Date",
        desc="Provide both the iOS version number and the exact release date for the basic Emergency SOS feature introduction",
        parent=feat_node,
        critical=True
    )
    ios_claim = (
        f"The basic Emergency SOS feature was first introduced in iOS {ios_ver}, released on {basic_date} (not the satellite version)."
    )
    await evaluator.verify(
        claim=ios_claim,
        node=ios_leaf,
        sources=_sources_or_none(data.basic_sources),
        additional_instruction=(
            "Verify the iOS version and exact release date for the initial Emergency SOS feature (non-satellite). "
            "Use Apple documentation or credible coverage. If unsupported or no URLs, mark Not Supported."
        ),
    )

    # Leaf: URL_Reference_Feature (presence/validity)
    url_feat_leaf = evaluator.add_leaf(
        id="URL_Reference_Feature",
        desc="Provide a valid URL confirming the iOS version and release date for Emergency SOS feature introduction",
        parent=feat_node,
        critical=True
    )
    url_feat_claim = "The provided URL(s) explicitly confirm the iOS version and release date for the basic Emergency SOS feature."
    await evaluator.verify(
        claim=url_feat_claim,
        node=url_feat_leaf,
        sources=_sources_or_none(data.basic_sources),
        additional_instruction=(
            "If no URL is provided or the URLs do not clearly confirm both version and date, mark Incorrect."
        ),
    )

    # Leaf: Satellite_Version_Date (source-backed)
    sat_date = data.satellite_launch_date or ""
    sat_leaf = evaluator.add_leaf(
        id="Satellite_Version_Date",
        desc="Identify the launch date for Emergency SOS via satellite feature on iPhone 14",
        parent=node,
        critical=True
    )
    sat_claim = f"Emergency SOS via satellite launched on {sat_date} for iPhone 14."
    await evaluator.verify(
        claim=sat_claim,
        node=sat_leaf,
        sources=_sources_or_none(data.satellite_sources),
        additional_instruction=(
            "Verify the satellite feature launch date using Apple announcements or credible reports. "
            "If unsupported or no URL provided, mark Not Supported."
        ),
    )

    # Leaf: Timeline_Gap_Calculation (logical check)
    gap_str = data.gap_years_months or ""
    gap_leaf = evaluator.add_leaf(
        id="Timeline_Gap_Calculation",
        desc="Calculate the approximate time gap between the basic Emergency SOS introduction and the satellite version launch, expressed in years and months",
        parent=node,
        critical=True
    )
    gap_claim = (
        f"The approximate time gap between {basic_date} (basic Emergency SOS introduction) and {sat_date} (satellite launch) "
        f"is {gap_str}."
    )
    await evaluator.verify(
        claim=gap_claim,
        node=gap_leaf,
        additional_instruction=(
            "Verify the time difference calculation (years and months). Minor rounding is acceptable if clearly approximate."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Verizon January 14, 2026 outage investigation task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root follows problem order (sequential)
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_investigation(),
        template_class=OutageInvestigationExtraction,
        extraction_name="investigation_extraction",
    )

    # Build verification tree according to rubric structure (critical sequential root)
    # Child 1: Outage Resolution Time
    await verify_outage_resolution(evaluator, root, extraction.outage_resolution)

    # Child 2: Technical Cause Identification
    await verify_technical_cause(evaluator, root, extraction.technical)

    # Child 3: Regulatory Compliance Deadline
    await verify_regulatory_deadline(evaluator, root, extraction.regulation)

    # Child 4: Related Emergency Feature Timeline
    await verify_emergency_feature(evaluator, root, extraction.emergency)

    # Optional: add custom info for transparency
    evaluator.add_custom_info(
        info={
            "policy_notes": [
                "Factual leaves are verified against cited URLs whenever available.",
                "When URLs are missing for source-backed claims, additional instructions enforce marking as Not Supported.",
                "Simple verify is used for pure logical checks (format validation, arithmetic date addition, time-gap calculation).",
            ]
        },
        info_type="evaluation_policy"
    )

    # Return summary
    return evaluator.get_summary()