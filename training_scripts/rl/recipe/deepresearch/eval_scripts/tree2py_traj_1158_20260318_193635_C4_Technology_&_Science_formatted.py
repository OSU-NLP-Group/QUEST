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
TASK_ID = "e911_infra_reqs"
TASK_DESCRIPTION = (
    "A telecommunications company is planning to upgrade its network infrastructure to support mission-critical "
    "911/E911 emergency services in compliance with current FCC regulations and industry standards. Based on federal "
    "requirements and carrier-grade standards, provide the following five specific requirements: "
    "(1) What is the minimum emergency backup power duration that telecommunications carriers must maintain for "
    "critical network assets under FCC regulations? "
    "(2) What is the maximum timeframe within which service providers must notify affected Public Safety Answering "
    "Points (PSAPs) after detecting a 911 service outage, according to FCC rules effective in 2025? "
    "(3) What is the minimum number of blocked calls (in a qualifying outage) that triggers mandatory FCC network "
    "outage reporting under 47 CFR 4.9? "
    "(4) What is the standard network availability level (expressed as a percentage and corresponding annual downtime) "
    "that carrier-grade telecommunications networks typically require to support mission-critical services? "
    "(5) What are the minimum data center tier classification(s) and their corresponding uptime percentages that should "
    "be used for hosting mission-critical telecommunications infrastructure?"
)

# Expected references and values used for judging (added to GT info for transparency)
EXPECTED_VALUES = {
    "backup_power_duration_min_hours": 24,  # 47 CFR Part 12 context
    "psap_notification_timeframe_minutes": 30,  # Effective Apr 15, 2025
    "psap_rule_effective_date": "April 15, 2025",
    "fcc_outage_reporting_blocked_calls_threshold": 90000,  # And >= 30 minutes duration
    "fcc_outage_reporting_min_duration_minutes": 30,
    "carrier_grade_availability_percent": "99.999%",
    "carrier_grade_downtime_minutes_per_year": 5.26,
    "tier_iii_uptime_percent": 99.982,
    "tier_iv_uptime_percent": 99.995,
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementItem(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EmergencyRequirementsExtraction(BaseModel):
    backup_power_duration: Optional[RequirementItem] = None
    psap_notification_timeframe: Optional[RequirementItem] = None
    fcc_outage_reporting_threshold: Optional[RequirementItem] = None
    network_availability_standard: Optional[RequirementItem] = None
    data_center_tier_requirement: Optional[RequirementItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract exactly what the answer states for each of the following five requirement items and the URLs it cites for each item.

    For each item, return:
    - value: A direct quote or faithful paraphrase of the requirement as stated in the answer (do not invent or normalize; keep the figure/wording the answer used, e.g., "24 hours", "30 minutes", "99.999% (five nines) ≈ 5.26 minutes/year", "Tier III 99.982% / Tier IV 99.995%").
    - sources: All URLs explicitly provided in the answer that support that specific item. Include only URLs that are actually present in the answer text. If no URL is provided for that item, return an empty array.

    Items to extract:
    1) backup_power_duration
       - Expect something like: "minimum 24 hours emergency backup power under 47 CFR Part 12", possibly with qualifiers like "central offices".
    2) psap_notification_timeframe
       - Expect something like: "notify affected PSAPs within 30 minutes of detecting a 911 outage", possibly noting that the rule is effective April 15, 2025.
    3) fcc_outage_reporting_threshold
       - Expect something like: "report outages that block at least 90,000 calls lasting 30 minutes or more" under 47 CFR 4.9.
    4) network_availability_standard
       - Expect something like: "carrier-grade = 99.999% (five nines), about 5.26 minutes/year".
    5) data_center_tier_requirement
       - Expect something like: "Tier III or Tier IV; Tier III 99.982% concurrent maintainability; Tier IV 99.995% 2N+1/fault tolerant".

    Special rules for source extraction:
    - Extract only URLs explicitly present in the answer. Accept plain URLs or markdown links; return normalized URLs.
    - Do not invent URLs.
    - If no URL is presented for an item, sources should be [].

    Return a JSON object conforming to the EmergencyRequirementsExtraction schema with these five top-level fields.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _has_value_and_sources(item: Optional[RequirementItem]) -> bool:
    return bool(item and item.value and item.value.strip() and item.sources and len(item.sources) > 0)


# --------------------------- Requirement #1 ---------------------------------#
async def verify_backup_power_requirement(
    evaluator: Evaluator,
    parent,
    ex: EmergencyRequirementsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Backup_Power_Duration",
        desc="The response must identify that telecommunications carriers must maintain emergency backup power for a minimum of 24 hours for critical network assets as required by 47 CFR Part 12",
        parent=parent,
        critical=False,
    )

    item = ex.backup_power_duration

    # Existence (critical)
    evaluator.add_custom_node(
        result=_has_value_and_sources(item),
        id="backup_power_exists",
        desc="Backup power requirement provided with at least one source URL",
        parent=node,
        critical=True,
    )

    # Value correctness (critical) - simple logical check against expected
    value_leaf = evaluator.add_leaf(
        id="backup_power_value_correct",
        desc="Backup power minimum duration stated as equivalent to 24 hours for critical assets",
        parent=node,
        critical=True,
    )
    stated = item.value if item and item.value else ""
    await evaluator.verify(
        claim=(
            f"The stated minimum emergency backup power duration for critical telecom network assets is equivalent to 24 hours. "
            f"The answer's extracted statement is: '{stated}'. Consider acceptable phrasings such as '24 hours', 'at least 24 hours', "
            f"'24h', or '24 hours for central offices' (treated as critical assets) as satisfying equivalence to 24 hours."
        ),
        node=value_leaf,
        additional_instruction="Focus on whether the stated value is semantically equivalent to a 24-hour minimum for critical assets."
    )

    # Source support (critical) - verify with cited URLs
    source_leaf = evaluator.add_leaf(
        id="backup_power_source_supported",
        desc="FCC Part 12 source supports minimum 24-hour emergency backup power for critical network assets",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="FCC regulations (47 CFR Part 12) require maintaining at least 24 hours of emergency backup power for critical network assets (e.g., central offices).",
        node=source_leaf,
        sources=item.sources if item else [],
        additional_instruction=(
            "Pass if the page(s) explicitly indicate a 24-hour emergency/standby/backup power minimum for central offices or critical telecom facilities under FCC Part 12. "
            "Allow equivalent wording. If the page discusses different durations for remote/field equipment, it should still indicate 24 hours for central offices or critical assets."
        ),
    )


# --------------------------- Requirement #2 ---------------------------------#
async def verify_psap_notification_requirement(
    evaluator: Evaluator,
    parent,
    ex: EmergencyRequirementsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="PSAP_Notification_Timeframe",
        desc="The response must identify that service providers must notify affected PSAPs within 30 minutes of detecting a 911 service outage, per FCC rules effective April 15, 2025",
        parent=parent,
        critical=False,
    )

    item = ex.psap_notification_timeframe

    # Existence (critical)
    evaluator.add_custom_node(
        result=_has_value_and_sources(item),
        id="psap_notify_exists",
        desc="PSAP notification requirement provided with at least one source URL",
        parent=node,
        critical=True,
    )

    # Value correctness - timeframe 30 minutes (critical)
    value_time_leaf = evaluator.add_leaf(
        id="psap_notify_value_30min",
        desc="PSAP notification timeframe stated as 30 minutes",
        parent=node,
        critical=True,
    )
    stated = item.value if item and item.value else ""
    await evaluator.verify(
        claim=(
            f"The extracted statement indicates that providers must notify affected PSAPs within 30 minutes of detecting/confirming a 911 service outage. "
            f"Extracted: '{stated}'. Accept equivalent wording such as 'no later than 30 minutes', 'within 30 minutes of discovery/detection/confirmation'."
        ),
        node=value_time_leaf,
    )

    # Value correctness - effective date in 2025-04-15 (critical)
    value_date_leaf = evaluator.add_leaf(
        id="psap_notify_value_effective_date",
        desc="PSAP notification rule effective date indicated as April 15, 2025 (or clearly 'effective in 2025' with that date)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The extracted statement indicates that this 30-minute PSAP notification rule is effective in 2025 (specifically April 15, 2025). "
            f"Extracted: '{stated}'. Accept close equivalents like 'effective 2025-04-15'."
        ),
        node=value_date_leaf,
    )

    # Source support - timeframe (critical)
    source_time_leaf = evaluator.add_leaf(
        id="psap_notify_source_time",
        desc="Sources support that PSAPs must be notified within 30 minutes of detecting a 911 service outage",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="FCC rules require service providers to notify affected PSAPs within 30 minutes of detecting a 911 service outage.",
        node=source_time_leaf,
        sources=item.sources if item else [],
        additional_instruction="Look for explicit 'within 30 minutes' language in the source materials."
    )

    # Source support - effective date (critical)
    source_date_leaf = evaluator.add_leaf(
        id="psap_notify_source_effective_date",
        desc="Sources indicate the effective date for the 30-minute PSAP notification rule is April 15, 2025",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The 30-minute PSAP notification requirement takes effect on April 15, 2025.",
        node=source_date_leaf,
        sources=item.sources if item else [],
        additional_instruction="Pass if the page specifies the effective date (April 15, 2025) or clearly indicates the rule becomes effective on that date."
    )


# --------------------------- Requirement #3 ---------------------------------#
async def verify_fcc_outage_reporting_requirement(
    evaluator: Evaluator,
    parent,
    ex: EmergencyRequirementsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="FCC_Outage_Reporting_Call_Threshold",
        desc="The response must identify that network outages blocking at least 90,000 calls (and lasting at least 30 minutes) must be reported to the FCC under 47 CFR 4.9",
        parent=parent,
        critical=False,
    )

    item = ex.fcc_outage_reporting_threshold

    # Existence (critical)
    evaluator.add_custom_node(
        result=_has_value_and_sources(item),
        id="outage_reporting_exists",
        desc="FCC outage reporting threshold provided with at least one source URL",
        parent=node,
        critical=True,
    )

    # Value correctness (critical)
    value_leaf = evaluator.add_leaf(
        id="outage_reporting_value_correct",
        desc="Outage reporting threshold includes 'at least 90,000 blocked calls' and 'at least 30 minutes' duration under 47 CFR 4.9",
        parent=node,
        critical=True,
    )
    stated = item.value if item and item.value else ""
    await evaluator.verify(
        claim=(
            f"The extracted statement correctly reflects that reportable outages under 47 CFR 4.9 include those that block at least 90,000 calls "
            f"and last at least 30 minutes. Extracted: '{stated}'. Both the 90,000-calls threshold and 30-minute duration must be present."
        ),
        node=value_leaf,
    )

    # Source support (critical)
    source_leaf = evaluator.add_leaf(
        id="outage_reporting_source_supported",
        desc="Sources support the 47 CFR 4.9 threshold (≥90,000 blocked calls and ≥30 minutes) for mandatory FCC outage reporting",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Under 47 CFR 4.9, providers must report outages that block at least 90,000 calls and last at least 30 minutes.",
        node=source_leaf,
        sources=item.sources if item else [],
        additional_instruction="Pass if the source explicitly mentions both the 90,000-call threshold and a duration threshold of at least 30 minutes."
    )


# --------------------------- Requirement #4 ---------------------------------#
async def verify_network_availability_requirement(
    evaluator: Evaluator,
    parent,
    ex: EmergencyRequirementsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Network_Availability_Standard",
        desc="The response must identify that carrier-grade telecommunications networks typically require 99.999% (five nines) availability or higher, which equals maximum 5.26 minutes downtime per year",
        parent=parent,
        critical=False,
    )

    item = ex.network_availability_standard

    # Existence (critical)
    evaluator.add_custom_node(
        result=_has_value_and_sources(item),
        id="availability_exists",
        desc="Carrier-grade availability figure provided with at least one source URL",
        parent=node,
        critical=True,
    )

    # Value correctness (critical)
    value_leaf = evaluator.add_leaf(
        id="availability_value_correct",
        desc="States 99.999% (five nines) availability and ~5.26 minutes annual downtime equivalence",
        parent=node,
        critical=True,
    )
    stated = item.value if item and item.value else ""
    await evaluator.verify(
        claim=(
            f"The extracted statement identifies a carrier-grade target of 99.999% (five nines) availability and its equivalence to about 5.26 minutes of downtime per year. "
            f"Extracted: '{stated}'. Minor rounding (e.g., ~5 minutes) is acceptable."
        ),
        node=value_leaf,
    )

    # Source support (critical)
    source_leaf = evaluator.add_leaf(
        id="availability_source_supported",
        desc="Sources support that carrier-grade networks target 99.999% availability (~5.26 minutes/year)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Carrier-grade telecommunications networks typically target at least 99.999% (five nines) availability, equating to roughly 5.26 minutes of downtime per year.",
        node=source_leaf,
        sources=item.sources if item else [],
        additional_instruction="Pass if the page explicitly references five nines (~99.999%) and the ~5.26 minutes/year equivalence or closely similar numbers."
    )


# --------------------------- Requirement #5 ---------------------------------#
async def verify_data_center_tier_requirement(
    evaluator: Evaluator,
    parent,
    ex: EmergencyRequirementsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Data_Center_Tier_Requirement",
        desc="The response must identify that Tier III or Tier IV data centers are required for mission-critical infrastructure, with Tier III providing 99.982% uptime through concurrent maintainability and Tier IV providing 99.995% uptime with 2N+1 redundancy",
        parent=parent,
        critical=False,
    )

    item = ex.data_center_tier_requirement

    # Existence (critical)
    evaluator.add_custom_node(
        result=_has_value_and_sources(item),
        id="dc_tier_exists",
        desc="Data center tier requirement provided with at least one source URL",
        parent=node,
        critical=True,
    )

    # Value correctness (critical)
    value_leaf = evaluator.add_leaf(
        id="dc_tier_value_correct",
        desc="States Tier III or IV usage; Tier III 99.982% concurrent maintainability; Tier IV 99.995% with 2N+1/fault tolerance",
        parent=node,
        critical=True,
    )
    stated = item.value if item and item.value else ""
    await evaluator.verify(
        claim=(
            f"The extracted statement identifies Tier III or Tier IV as appropriate for mission-critical infrastructure and includes that "
            f"Tier III provides 99.982% uptime (concurrently maintainable) and Tier IV provides 99.995% uptime with 2N+1 redundancy (fault tolerant). "
            f"Extracted: '{stated}'. Accept clear equivalents (e.g., 'concurrent maintainability' wording variations; '2N+1' and/or 'fault tolerant')."
        ),
        node=value_leaf,
    )

    # Source support - Tier III (critical)
    source_t3_leaf = evaluator.add_leaf(
        id="dc_tier_source_tier3",
        desc="Sources support Tier III = 99.982% uptime and concurrent maintainability",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Uptime Institute Tier III data centers provide approximately 99.982% uptime and are concurrently maintainable.",
        node=source_t3_leaf,
        sources=item.sources if item else [],
        additional_instruction="Pass if the page clearly states 99.982% uptime for Tier III and mentions concurrent maintainability (or clear equivalent)."
    )

    # Source support - Tier IV (critical)
    source_t4_leaf = evaluator.add_leaf(
        id="dc_tier_source_tier4",
        desc="Sources support Tier IV = 99.995% uptime and 2N+1 redundancy (fault tolerance)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Uptime Institute Tier IV data centers provide approximately 99.995% uptime and 2N+1 redundancy, offering fault tolerance.",
        node=source_t4_leaf,
        sources=item.sources if item else [],
        additional_instruction="Pass if the page clearly states 99.995% uptime for Tier IV and describes 2N+1 redundancy and/or fault tolerance."
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
    Evaluate an answer for the emergency services infrastructure requirements task.
    """
    # Initialize evaluator; use parallel root to allow partial credit across the five requirements
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

    # Record expected ground truth info
    evaluator.add_ground_truth(
        {
            "expected_values": EXPECTED_VALUES,
            "notes": "These represent commonly cited federal/industry thresholds used for judging. Source-grounded verification is performed against the answer's cited URLs."
        }
    )

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=EmergencyRequirementsExtraction,
        extraction_name="requirements_extracted",
    )

    # Build verification subtree corresponding to the rubric
    # The top-level rubric node is represented by the root created above.
    await verify_backup_power_requirement(evaluator, root, extracted)
    await verify_psap_notification_requirement(evaluator, root, extracted)
    await verify_fcc_outage_reporting_requirement(evaluator, root, extracted)
    await verify_network_availability_requirement(evaluator, root, extracted)
    await verify_data_center_tier_requirement(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()