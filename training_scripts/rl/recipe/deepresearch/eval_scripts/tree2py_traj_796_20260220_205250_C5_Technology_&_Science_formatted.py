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
TASK_ID = "telecom_outage_policies"
TASK_DESCRIPTION = (
    "A telecommunications regulatory compliance team is preparing a comprehensive report on US network outage "
    "requirements and carrier consumer protection policies. They need to gather the following specific information: "
    "(1) What are the FCC-mandated timelines for telecommunications service providers to submit (a) the initial outage "
    "report and (b) the final outage report after discovering a qualifying network outage? "
    "(2) What is the FCC-mandated minimum number of hours of backup power required for cell tower sites, and what is the "
    "range of extended backup power hours required for certain high-risk areas? "
    "(3) What are AT&T's specific compensation policy thresholds for service outages, including: (a) the minimum downtime "
    "duration in minutes that qualifies for credit for wired services, (b) the minimum downtime duration in minutes for "
    "wireless services, and (c) the minimum number of towers that must be affected for wireless compensation to apply? "
    "(4) What dollar amount of account credit did Verizon offer to affected customers after its major outage in January 2026, "
    "and on what specific date did this outage occur? "
    "(5) What are the FCC requirements for voice service providers regarding 911/988 outage notifications, including: "
    "(a) the timeline in minutes for initial notification to affected PSAPs after discovering the outage, and "
    "(b) the required frequency in hours for follow-up notifications during an ongoing outage? "
    "Provide all numerical values and dates, along with reference URLs that support each piece of information."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FCCReportingExtraction(BaseModel):
    initial_report_timeline: Optional[str] = None
    final_report_timeline: Optional[str] = None
    reporting_sources: List[str] = Field(default_factory=list)


class BackupPowerExtraction(BaseModel):
    minimum_backup_hours: Optional[str] = None
    extended_backup_range: Optional[str] = None
    backup_sources: List[str] = Field(default_factory=list)


class ATTCompensationExtraction(BaseModel):
    wired_threshold_minutes: Optional[str] = None
    wireless_threshold_minutes: Optional[str] = None
    wireless_tower_requirement_count: Optional[str] = None
    policy_sources: List[str] = Field(default_factory=list)


class VerizonCompensationExtraction(BaseModel):
    credit_amount: Optional[str] = None
    outage_date: Optional[str] = None
    credit_sources: List[str] = Field(default_factory=list)


class EmergencyNotificationExtraction(BaseModel):
    initial_psap_notification_minutes: Optional[str] = None
    followup_notification_frequency_hours: Optional[str] = None
    notification_sources: List[str] = Field(default_factory=list)


class CombinedExtraction(BaseModel):
    fcc_reporting: Optional[FCCReportingExtraction] = None
    backup_power: Optional[BackupPowerExtraction] = None
    att_policy: Optional[ATTCompensationExtraction] = None
    verizon_policy: Optional[VerizonCompensationExtraction] = None
    emergency_notification: Optional[EmergencyNotificationExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_combined() -> str:
    return """
    Extract the following information exactly as presented in the answer text. Preserve units for durations (e.g., "minutes", "hours", "days"), include dollar signs for money amounts if present (e.g., "$10"), and use complete dates (e.g., "January 25, 2026").

    Return a single JSON object with these nested sections and fields:

    fcc_reporting:
      - initial_report_timeline: string | null
      - final_report_timeline: string | null
      - reporting_sources: array of URLs (explicitly cited in the answer; may be markdown links) — empty array if none

    backup_power:
      - minimum_backup_hours: string | null
      - extended_backup_range: string | null    # e.g., "24–72 hours" or "24 to 72 hours"
      - backup_sources: array of URLs — empty array if none

    att_policy:
      - wired_threshold_minutes: string | null
      - wireless_threshold_minutes: string | null
      - wireless_tower_requirement_count: string | null
      - policy_sources: array of URLs — empty array if none

    verizon_policy:
      - credit_amount: string | null            # e.g., "$10" or "10 dollars"
      - outage_date: string | null              # e.g., "January 25, 2026"
      - credit_sources: array of URLs — empty array if none

    emergency_notification:
      - initial_psap_notification_minutes: string | null
      - followup_notification_frequency_hours: string | null
      - notification_sources: array of URLs — empty array if none

    Rules:
    - Extract ONLY what is explicitly present in the answer; do not infer or invent.
    - If a field is missing, set it to null; if sources are not provided for a section, return an empty array.
    - Extract actual URLs for sources (including markdown link targets).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_present(val: Optional[str]) -> bool:
    return bool(val) and bool(str(val).strip())


def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def _build_instruction_require_sources(value_present: bool, sources_present: bool, base: str) -> str:
    """
    Build an instruction that enforces source-grounding and value presence.
    """
    parts = [base.strip()]
    parts.append("Always rely on explicit evidence from the provided webpage(s). Allow minor formatting variations (e.g., '$10' vs '$10.00', '24–72' vs '24 to 72').")
    if not sources_present:
        parts.append("No source URLs are provided; you must mark this claim as NOT SUPPORTED.")
    if not value_present:
        parts.append("The value to be verified is missing or blank in the answer; you must mark this claim as NOT SUPPORTED.")
    return " ".join(parts)


def _safe_sources(urls: Optional[List[str]]) -> Optional[List[str]]:
    """
    Normalize sources list; return None if empty (routes to simple verification).
    """
    if not urls:
        return None
    cleaned = [u for u in urls if isinstance(u, str) and u.strip()]
    return cleaned or None


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_fcc_reporting(evaluator: Evaluator, parent_node, data: Optional[FCCReportingExtraction]) -> None:
    node = evaluator.add_parallel(
        id="fcc_reporting_requirements",
        desc="Identify FCC outage reporting timeline requirements for telecommunications service providers",
        parent=parent_node,
        critical=False
    )

    init_val = data.initial_report_timeline if data else None
    final_val = data.final_report_timeline if data else None
    sources = _safe_sources(data.reporting_sources if data else [])

    # Initial report timeline
    init_leaf = evaluator.add_leaf(
        id="initial_report_timeline",
        desc="Provide the FCC-mandated timeline for submitting the initial outage report after discovery",
        parent=node,
        critical=True
    )
    init_claim = f"The FCC-mandated timeline for submitting the initial outage report after discovering a qualifying outage is {init_val}."
    init_ins = _build_instruction_require_sources(
        value_present=_is_present(init_val),
        sources_present=_has_sources(sources),
        base="Verify the initial reporting deadline using the cited FCC rule, ECFR Part 4, or official FCC documentation."
    )
    await evaluator.verify(
        claim=init_claim,
        node=init_leaf,
        sources=sources,
        additional_instruction=init_ins
    )

    # Final report timeline
    final_leaf = evaluator.add_leaf(
        id="final_report_timeline",
        desc="Provide the FCC-mandated timeline for submitting the final outage report after discovery",
        parent=node,
        critical=True
    )
    final_claim = f"The FCC-mandated timeline for submitting the final outage report after discovering a qualifying outage is {final_val}."
    final_ins = _build_instruction_require_sources(
        value_present=_is_present(final_val),
        sources_present=_has_sources(sources),
        base="Verify the final reporting deadline using the cited FCC rule, ECFR Part 4, or official FCC documentation."
    )
    await evaluator.verify(
        claim=final_claim,
        node=final_leaf,
        sources=sources,
        additional_instruction=final_ins
    )

    # Reference verification
    ref_leaf = evaluator.add_leaf(
        id="fcc_reporting_reference",
        desc="Provide a valid reference URL supporting the FCC reporting timeline requirements",
        parent=node,
        critical=True
    )
    ref_claim = "These sources explicitly discuss FCC outage reporting timeline requirements, including deadlines for both initial and final reports."
    ref_ins = _build_instruction_require_sources(
        value_present=True,
        sources_present=_has_sources(sources),
        base="Confirm that at least one cited source is an official FCC/ECFR or credible document that clearly states outage reporting timelines."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=sources,
        additional_instruction=ref_ins
    )


async def verify_backup_power(evaluator: Evaluator, parent_node, data: Optional[BackupPowerExtraction]) -> None:
    node = evaluator.add_parallel(
        id="backup_power_standards",
        desc="Identify FCC backup power requirements for cell tower sites",
        parent=parent_node,
        critical=False
    )

    min_val = data.minimum_backup_hours if data else None
    range_val = data.extended_backup_range if data else None
    sources = _safe_sources(data.backup_sources if data else [])

    # Minimum backup hours
    min_leaf = evaluator.add_leaf(
        id="minimum_backup_hours",
        desc="Provide the FCC-mandated minimum hours of backup power required for cell sites",
        parent=node,
        critical=True
    )
    min_claim = f"The FCC-mandated minimum backup power duration for cell sites is {min_val}."
    min_ins = _build_instruction_require_sources(
        value_present=_is_present(min_val),
        sources_present=_has_sources(sources),
        base="Verify the minimum backup power duration using FCC orders, rules, or official resilience/backup power policies."
    )
    await evaluator.verify(
        claim=min_claim,
        node=min_leaf,
        sources=sources,
        additional_instruction=min_ins
    )

    # Extended backup range
    ext_leaf = evaluator.add_leaf(
        id="extended_backup_range",
        desc="Provide the range of extended backup power hours required for certain high-risk areas",
        parent=node,
        critical=True
    )
    ext_claim = f"For certain high-risk areas, the extended backup power requirement range is {range_val}."
    ext_ins = _build_instruction_require_sources(
        value_present=_is_present(range_val),
        sources_present=_has_sources(sources),
        base="Verify the extended backup power range from FCC or official sources describing high-risk or disaster-prone area requirements."
    )
    await evaluator.verify(
        claim=ext_claim,
        node=ext_leaf,
        sources=sources,
        additional_instruction=ext_ins
    )

    # Reference verification
    ref_leaf = evaluator.add_leaf(
        id="backup_power_reference",
        desc="Provide a valid reference URL supporting the backup power requirements",
        parent=node,
        critical=True
    )
    ref_claim = "These sources explicitly state FCC backup power requirements for cell tower sites, including minimum and any extended durations."
    ref_ins = _build_instruction_require_sources(
        value_present=True,
        sources_present=_has_sources(sources),
        base="Confirm at least one cited source is an official FCC document or regulation that clearly describes backup power requirements."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=sources,
        additional_instruction=ref_ins
    )


async def verify_att_policy(evaluator: Evaluator, parent_node, data: Optional[ATTCompensationExtraction]) -> None:
    node = evaluator.add_parallel(
        id="att_compensation_policy",
        desc="Identify AT&T's compensation policies for service outages",
        parent=parent_node,
        critical=False
    )

    wired = data.wired_threshold_minutes if data else None
    wireless = data.wireless_threshold_minutes if data else None
    towers = data.wireless_tower_requirement_count if data else None
    sources = _safe_sources(data.policy_sources if data else [])

    # Wired threshold
    wired_leaf = evaluator.add_leaf(
        id="att_wired_threshold",
        desc="Provide the minimum downtime duration (in minutes) that qualifies for AT&T credit for wired services",
        parent=node,
        critical=True
    )
    wired_claim = f"AT&T provides credit for wired services when downtime lasts at least {wired} minutes."
    wired_ins = _build_instruction_require_sources(
        value_present=_is_present(wired),
        sources_present=_has_sources(sources),
        base="Verify the wired outage credit threshold using AT&T official policy pages or terms of service."
    )
    await evaluator.verify(
        claim=wired_claim,
        node=wired_leaf,
        sources=sources,
        additional_instruction=wired_ins
    )

    # Wireless threshold
    wireless_leaf = evaluator.add_leaf(
        id="att_wireless_threshold",
        desc="Provide the minimum downtime duration (in minutes) that qualifies for AT&T credit for wireless services",
        parent=node,
        critical=True
    )
    wireless_claim = f"AT&T provides credit for wireless services when downtime lasts at least {wireless} minutes."
    wireless_ins = _build_instruction_require_sources(
        value_present=_is_present(wireless),
        sources_present=_has_sources(sources),
        base="Verify the wireless outage credit threshold using AT&T official policy pages or terms of service."
    )
    await evaluator.verify(
        claim=wireless_claim,
        node=wireless_leaf,
        sources=sources,
        additional_instruction=wireless_ins
    )

    # Wireless tower requirement
    towers_leaf = evaluator.add_leaf(
        id="att_wireless_tower_requirement",
        desc="Provide the minimum number of towers that must be affected for AT&T wireless compensation to apply",
        parent=node,
        critical=True
    )
    towers_claim = f"AT&T wireless compensation applies only if at least {towers} towers are affected."
    towers_ins = _build_instruction_require_sources(
        value_present=_is_present(towers),
        sources_present=_has_sources(sources),
        base="Verify the tower-count condition from AT&T official compensation policy or service credit terms."
    )
    await evaluator.verify(
        claim=towers_claim,
        node=towers_leaf,
        sources=sources,
        additional_instruction=towers_ins
    )

    # Reference verification
    ref_leaf = evaluator.add_leaf(
        id="att_policy_reference",
        desc="Provide a valid reference URL supporting AT&T's compensation policy",
        parent=node,
        critical=True
    )
    ref_claim = "These sources are official AT&T policy or credible documents that explicitly state outage compensation thresholds and conditions."
    ref_ins = _build_instruction_require_sources(
        value_present=True,
        sources_present=_has_sources(sources),
        base="Confirm at least one cited source is AT&T official content or equivalent authoritative documentation."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=sources,
        additional_instruction=ref_ins
    )


async def verify_verizon_policy(evaluator: Evaluator, parent_node, data: Optional[VerizonCompensationExtraction]) -> None:
    node = evaluator.add_parallel(
        id="verizon_compensation_policy",
        desc="Identify Verizon's compensation offered after the January 2026 outage",
        parent=parent_node,
        critical=False
    )

    amount = data.credit_amount if data else None
    outage_date = data.outage_date if data else None
    sources = _safe_sources(data.credit_sources if data else [])

    # Credit amount
    amount_leaf = evaluator.add_leaf(
        id="verizon_credit_amount",
        desc="Provide the dollar amount of account credit Verizon offered to affected customers after the January 2026 outage",
        parent=node,
        critical=True
    )
    amount_claim = f"Following the January 2026 outage, Verizon offered an account credit of {amount} to affected customers."
    amount_ins = _build_instruction_require_sources(
        value_present=_is_present(amount),
        sources_present=_has_sources(sources),
        base="Verify the exact credit amount from Verizon's official communications or credible reports."
    )
    await evaluator.verify(
        claim=amount_claim,
        node=amount_leaf,
        sources=sources,
        additional_instruction=amount_ins
    )

    # Outage date
    date_leaf = evaluator.add_leaf(
        id="verizon_outage_date",
        desc="Provide the specific date in January 2026 when the major Verizon outage occurred",
        parent=node,
        critical=True
    )
    date_claim = f"The major Verizon outage occurred on {outage_date} in January 2026."
    date_ins = _build_instruction_require_sources(
        value_present=_is_present(outage_date),
        sources_present=_has_sources(sources),
        base="Verify the specific outage date from Verizon's official communications or credible, well-sourced news."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=sources,
        additional_instruction=date_ins
    )

    # Reference verification
    ref_leaf = evaluator.add_leaf(
        id="verizon_credit_reference",
        desc="Provide a valid reference URL supporting Verizon's compensation offer",
        parent=node,
        critical=True
    )
    ref_claim = "These sources explicitly state Verizon's compensation offer and the outage date for the January 2026 incident."
    ref_ins = _build_instruction_require_sources(
        value_present=True,
        sources_present=_has_sources(sources),
        base="Confirm at least one cited source clearly states the compensation amount and outage date."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=sources,
        additional_instruction=ref_ins
    )


async def verify_emergency_notification(evaluator: Evaluator, parent_node, data: Optional[EmergencyNotificationExtraction]) -> None:
    node = evaluator.add_parallel(
        id="emergency_notification_requirements",
        desc="Identify FCC requirements for 911/988 outage notifications to PSAPs",
        parent=parent_node,
        critical=False
    )

    init_minutes = data.initial_psap_notification_minutes if data else None
    follow_hours = data.followup_notification_frequency_hours if data else None
    sources = _safe_sources(data.notification_sources if data else [])

    # Initial notification timeline
    init_leaf = evaluator.add_leaf(
        id="initial_notification_timeline",
        desc="Provide the FCC-mandated timeline (in minutes) for notifying PSAPs of a 911/988-impacting outage after discovery",
        parent=node,
        critical=True
    )
    init_claim = f"Voice service providers must notify affected PSAPs within {init_minutes} after discovering a 911/988-impacting outage."
    init_ins = _build_instruction_require_sources(
        value_present=_is_present(init_minutes),
        sources_present=_has_sources(sources),
        base="Verify the initial notification timeline from FCC rules, orders, or official guidance; ensure the unit is minutes."
    )
    await evaluator.verify(
        claim=init_claim,
        node=init_leaf,
        sources=sources,
        additional_instruction=init_ins
    )

    # Follow-up notification frequency
    follow_leaf = evaluator.add_leaf(
        id="followup_notification_frequency",
        desc="Provide the required frequency (in hours) for follow-up notifications to PSAPs during an ongoing outage",
        parent=node,
        critical=True
    )
    follow_claim = f"During an ongoing outage, providers must send follow-up notifications to affected PSAPs every {follow_hours}."
    follow_ins = _build_instruction_require_sources(
        value_present=_is_present(follow_hours),
        sources_present=_has_sources(sources),
        base="Verify the follow-up notification frequency from FCC rules, orders, or official guidance; ensure the unit is hours."
    )
    await evaluator.verify(
        claim=follow_claim,
        node=follow_leaf,
        sources=sources,
        additional_instruction=follow_ins
    )

    # Reference verification
    ref_leaf = evaluator.add_leaf(
        id="emergency_notification_reference",
        desc="Provide a valid reference URL supporting the 911/988 notification requirements",
        parent=node,
        critical=True
    )
    ref_claim = "These sources explicitly state FCC requirements for PSAP outage notifications, including initial timeline and follow-up frequency."
    ref_ins = _build_instruction_require_sources(
        value_present=True,
        sources_present=_has_sources(sources),
        base="Confirm at least one cited source clearly states both the initial PSAP notification timeframe and the follow-up cadence."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=sources,
        additional_instruction=ref_ins
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
    Evaluate an answer for the telecommunications outage reporting, backup power, and carrier policy task.
    """
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

    # Extract all required info in one pass
    extracted = await evaluator.extract(
        prompt=prompt_extract_combined(),
        template_class=CombinedExtraction,
        extraction_name="telecom_outage_compilation"
    )

    # Build tree for each major section (parallel under root)
    await verify_fcc_reporting(evaluator, root, extracted.fcc_reporting)
    await verify_backup_power(evaluator, root, extracted.backup_power)

    carrier_node = evaluator.add_parallel(
        id="carrier_compensation_policies",
        desc="Identify customer compensation policies from two major US carriers for service outages",
        parent=root,
        critical=False
    )
    await verify_att_policy(evaluator, carrier_node, extracted.att_policy)
    await verify_verizon_policy(evaluator, carrier_node, extracted.verizon_policy)

    await verify_emergency_notification(evaluator, root, extracted.emergency_notification)

    return evaluator.get_summary()