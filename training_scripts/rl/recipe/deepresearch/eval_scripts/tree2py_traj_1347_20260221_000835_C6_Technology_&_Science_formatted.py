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
TASK_ID = "telecom_outage_compliance_2026"
TASK_DESCRIPTION = (
    "A wireless telecommunications provider in Texas experiences a network outage affecting 12 cell sites on "
    "February 15, 2026. Each cell site normally serves an average of 1,300 subscribers. The outage begins at 2:00 PM "
    "local time and full service is restored at 8:00 PM (6 hours = 360 minutes total duration). The provider's backup "
    "battery systems at the affected cell sites lasted 5 hours before complete depletion. The provider had committed "
    "to achieving 99.99% network availability in their service level agreement.\n\n"
    "Based on FCC regulations and telecommunications industry standards, provide a comprehensive analysis determining:\n"
    "1. Whether this outage meets the FCC NORS reporting thresholds and must be reported\n"
    "2. The specific FCC reporting timeline requirements that apply (initial notification, initial report, final report deadlines)\n"
    "3. Whether the backup power systems were compliant with FCC requirements for cell sites\n"
    "4. The impact on annual network availability and whether the SLA commitment would be violated if this were the only outage for the year\n"
    "5. What network redundancy standards (N+1 and geographic diversity) should have been in place to prevent or mitigate this type of outage\n\n"
    "For each determination, provide the specific regulatory or industry standard that applies, the calculation or analysis performed, and the conclusion. "
    "All determinations must be supported by reference to FCC regulations, industry standards, or authoritative telecommunications infrastructure requirements."
)

# Scenario constants and derived values
SCENARIO_CELL_SITES = 12
SCENARIO_USERS_PER_SITE = 1300
OUTAGE_DURATION_MINUTES = 360
TOTAL_USER_MINUTES = SCENARIO_CELL_SITES * SCENARIO_USERS_PER_SITE * OUTAGE_DURATION_MINUTES  # 5,616,000
BACKUP_REQUIREMENT_MIN_HOURS = 8
ACTUAL_BACKUP_HOURS = 5
YEAR_MINUTES = 365 * 24 * 60  # 525,600
ALLOWABLE_DOWNTIME_99_99_MINUTES = YEAR_MINUTES * 0.0001  # 52.56 minutes
CALCULATED_AVAILABILITY_PERCENT = 100.0 * (1.0 - (OUTAGE_DURATION_MINUTES / YEAR_MINUTES))  # ≈ 99.9314%

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NORSCalculation(BaseModel):
    cell_sites_affected: Optional[str] = None
    users_per_site_avg: Optional[str] = None
    outage_duration_minutes: Optional[str] = None
    total_user_minutes: Optional[str] = None
    user_minutes_reference_urls: List[str] = Field(default_factory=list)
    threshold_reference_urls: List[str] = Field(default_factory=list)


class NORSTimeline(BaseModel):
    initial_notification_minutes: Optional[str] = None
    initial_report_timeframe: Optional[str] = None  # e.g., "72 hours" or "3 days"
    final_report_days: Optional[str] = None
    timeline_reference_urls: List[str] = Field(default_factory=list)


class BackupPowerInfo(BaseModel):
    fcc_backup_requirement_hours: Optional[str] = None
    backup_requirement_reference_urls: List[str] = Field(default_factory=list)
    actual_backup_duration_hours: Optional[str] = None
    compliance_reference_urls: List[str] = Field(default_factory=list)


class AvailabilityInfo(BaseModel):
    downtime_to_annual_calc_text: Optional[str] = None
    calculated_availability_percent: Optional[str] = None
    availability_calc_reference_urls: List[str] = Field(default_factory=list)
    sla_commitment_percent: Optional[str] = None
    allowable_downtime_minutes: Optional[str] = None
    standards_reference_urls: List[str] = Field(default_factory=list)


class RedundancyInfo(BaseModel):
    n_plus_one_definition_text: Optional[str] = None
    n_plus_one_reference_urls: List[str] = Field(default_factory=list)
    geographic_diversity_definition_text: Optional[str] = None
    geographic_diversity_reference_urls: List[str] = Field(default_factory=list)
    redundancy_purpose_text: Optional[str] = None


class OutageAnalysisExtraction(BaseModel):
    nors_calc: Optional[NORSCalculation] = None
    timeline: Optional[NORSTimeline] = None
    backup: Optional[BackupPowerInfo] = None
    availability: Optional[AvailabilityInfo] = None
    redundancy: Optional[RedundancyInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_analysis() -> str:
    return """
    Extract the specific structured information that the answer presents for the telecommunications outage analysis. 
    Only extract what is explicitly stated in the answer. If the answer does not provide any item, return null or an empty list for that item.

    For 'nors_calc', extract:
    - cell_sites_affected: the number of cell sites affected (as written)
    - users_per_site_avg: the average subscribers per cell site (as written)
    - outage_duration_minutes: the outage duration stated in minutes (if stated; if only hours are given, extract that text)
    - total_user_minutes: the total user-minutes for the outage (as written in the answer, if provided)
    - user_minutes_reference_urls: all URLs cited that explain FCC's NORS user-minutes calculation methodology
    - threshold_reference_urls: all URLs cited that explain FCC reporting thresholds (e.g., 47 CFR Part 4 / §4.9, NORS docs)

    For 'timeline', extract:
    - initial_notification_minutes: the initial notification requirement value (e.g., "120 minutes") as written
    - initial_report_timeframe: the initial report deadline (e.g., "72 hours" or "3 days") as written
    - final_report_days: the final report deadline (e.g., "30 days") as written
    - timeline_reference_urls: all URLs cited that support the timeline requirements

    For 'backup', extract:
    - fcc_backup_requirement_hours: the stated FCC minimum backup duration for cell sites (e.g., "8 hours")
    - backup_requirement_reference_urls: URLs cited that support the FCC backup power requirement
    - actual_backup_duration_hours: the actual backup power duration from the scenario (as written in the answer)
    - compliance_reference_urls: URLs cited that support the compliance determination criteria

    For 'availability', extract:
    - downtime_to_annual_calc_text: any text that shows converting downtime to annual proportion (as written)
    - calculated_availability_percent: the availability percentage the answer calculates for this outage (as written, e.g., "99.93%")
    - availability_calc_reference_urls: URLs cited that explain availability calculation methodology (e.g., formula)
    - sla_commitment_percent: the SLA commitment percentage identified (as written, e.g., "99.99%")
    - allowable_downtime_minutes: the allowable downtime for 99.99% per year (as written, e.g., "~52.56 minutes")
    - standards_reference_urls: URLs cited that define or tabulate downtime equivalents for availability targets

    For 'redundancy', extract:
    - n_plus_one_definition_text: the definition/explanation of N+1 redundancy from the answer
    - n_plus_one_reference_urls: URLs cited that define N+1 redundancy
    - geographic_diversity_definition_text: the definition/explanation of geographic diversity / diverse routing from the answer
    - geographic_diversity_reference_urls: URLs cited that define geographic diversity / diverse routing
    - redundancy_purpose_text: the explanation of how N+1 and geographic diversity mitigate multi-site outages (as written)

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer (including markdown link targets). If a source is mentioned without a URL, do not invent one; return an empty list.
    - Always include full URLs with protocol if available. Ignore obviously malformed URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls or []


def fmt_pct(p: float, decimals: int = 3) -> str:
    return f"{round(p, decimals)}%"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_nors_reporting(evaluator: Evaluator, parent_node, data: OutageAnalysisExtraction) -> None:
    nors = data.nors_calc or NORSCalculation()

    nors_root = evaluator.add_sequential(
        id="FCC_NORS_Reporting_Determination",
        desc="Determine whether the outage meets FCC NORS reporting thresholds and must be reported to the Commission",
        parent=parent_node,
        critical=False
    )

    # 1) User_Minutes_Calculation (parallel, critical)
    um_node = evaluator.add_parallel(
        id="User_Minutes_Calculation",
        desc="Calculate total user-minutes affected by the outage using FCC methodology",
        parent=nors_root,
        critical=True
    )

    # 1.1 Cell Sites Affected
    leaf_sites = evaluator.add_leaf(
        id="Cell_Sites_Affected",
        desc="Correctly identify the number of cell sites affected by the outage from the scenario",
        parent=um_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer identifies that the outage affected {SCENARIO_CELL_SITES} cell sites.",
        node=leaf_sites,
        additional_instruction="Judge correct if the answer clearly states 12 affected cell sites (accept 'dozen' as equivalent)."
    )

    # 1.2 Average Users per Site
    leaf_users = evaluator.add_leaf(
        id="Average_Users_Per_Site",
        desc="Correctly apply the average users per cell site from the scenario",
        parent=um_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Each affected cell site serves an average of {SCENARIO_USERS_PER_SITE} subscribers.",
        node=leaf_users,
        additional_instruction="Judge correct if the answer clearly cites 1,300 subscribers per site."
    )

    # 1.3 Outage Duration Minutes
    leaf_duration = evaluator.add_leaf(
        id="Outage_Duration_Minutes",
        desc="Correctly convert the outage duration to minutes",
        parent=um_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The outage lasted {OUTAGE_DURATION_MINUTES} minutes (6 hours).",
        node=leaf_duration,
        additional_instruction="Allow mentioning '6 hours' as equivalent to 360 minutes; treat them as the same duration."
    )

    # 1.4 Total User Minutes Calculation
    leaf_total_um = evaluator.add_leaf(
        id="Total_User_Minutes_Calculation",
        desc="Correctly calculate total user-minutes by multiplying cell sites × users per site × duration in minutes",
        parent=um_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total user‑minutes affected equals {TOTAL_USER_MINUTES}.",
        node=leaf_total_um,
        additional_instruction="Judge correct if the answer explicitly presents the calculation 12×1300×360 and/or the result 5,616,000 user‑minutes."
    )

    # 1.5 Calculation Reference
    leaf_um_ref = evaluator.add_leaf(
        id="Calculation_Reference",
        desc="Provide reference to FCC NORS methodology for user-minutes calculation",
        parent=um_node,
        critical=True
    )
    await evaluator.verify(
        claim="FCC NORS methodology defines 'user‑minutes' as the number of users affected multiplied by the outage duration in minutes.",
        node=leaf_um_ref,
        sources=safe_urls(nors.user_minutes_reference_urls),
        additional_instruction="Verify the referenced FCC/NORS documentation explicitly describes the user‑minutes calculation method."
    )

    # 2) Reportability_Threshold_Assessment (parallel, critical)
    rta_node = evaluator.add_parallel(
        id="Reportability_Threshold_Assessment",
        desc="Evaluate whether the calculated user-minutes and duration meet FCC reporting thresholds",
        parent=nors_root,
        critical=True
    )

    # 2.1 User Minutes Threshold
    leaf_um_thresh = evaluator.add_leaf(
        id="User_Minutes_Threshold",
        desc="Correctly determine whether the calculated user-minutes exceeds the 900,000 user-minute FCC threshold",
        parent=rta_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{TOTAL_USER_MINUTES} user‑minutes exceeds the FCC 900,000 user‑minutes reporting threshold.",
        node=leaf_um_thresh,
        sources=safe_urls(nors.threshold_reference_urls),
        additional_instruction="Use the cited FCC/NORS threshold (900,000 user‑minutes) to judge this statement."
    )

    # 2.2 Duration Threshold
    leaf_dur_thresh = evaluator.add_leaf(
        id="Duration_Threshold",
        desc="Correctly determine whether the outage duration exceeds the 30-minute minimum threshold",
        parent=rta_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The outage duration of {OUTAGE_DURATION_MINUTES} minutes exceeds the FCC minimum 30‑minute threshold.",
        node=leaf_dur_thresh,
        sources=safe_urls(nors.threshold_reference_urls),
        additional_instruction="Use the cited FCC/NORS minimum duration threshold (30 minutes) to judge this statement."
    )

    # 2.3 Reportability Conclusion
    leaf_reportable = evaluator.add_leaf(
        id="Reportability_Conclusion",
        desc="Correctly conclude whether the outage is FCC NORS reportable based on threshold analysis",
        parent=rta_node,
        critical=True
    )
    await evaluator.verify(
        claim="This outage meets FCC NORS reporting thresholds and must be reported to the FCC.",
        node=leaf_reportable,
        sources=safe_urls(nors.threshold_reference_urls),
        additional_instruction="Base your decision on exceeding both the 900,000 user‑minutes and 30‑minute duration thresholds."
    )

    # 2.4 Threshold Reference
    leaf_thresh_ref = evaluator.add_leaf(
        id="Threshold_Reference",
        desc="Provide reference to 47 CFR §4.9 or FCC NORS documentation",
        parent=rta_node,
        critical=True
    )
    await evaluator.verify(
        claim="The cited sources are authoritative FCC Part 4/§4.9 or NORS documentation that define outage reporting thresholds.",
        node=leaf_thresh_ref,
        sources=safe_urls(nors.threshold_reference_urls),
        additional_instruction="Confirm the sources explicitly correspond to FCC outage reporting rules (e.g., 47 CFR Part 4) or official NORS guidance."
    )

    # 3) Reporting_Timeline_Requirements (parallel, non-critical)
    tl_node = evaluator.add_parallel(
        id="Reporting_Timeline_Requirements",
        desc="Identify the specific FCC reporting deadlines that apply if the outage is reportable",
        parent=nors_root,
        critical=False
    )

    leaf_initial_notify = evaluator.add_leaf(
        id="Initial_Notification_Requirement",
        desc="Correctly identify the 120-minute initial notification requirement for wireless carriers",
        parent=tl_node,
        critical=True
    )
    await evaluator.verify(
        claim="Wireless providers must submit an initial NORS notification within 120 minutes of discovering a reportable outage.",
        node=leaf_initial_notify,
        sources=safe_urls((data.timeline or NORSTimeline()).timeline_reference_urls),
        additional_instruction="Verify the 120‑minute initial notification requirement from the cited FCC/NORS source."
    )

    leaf_initial_report = evaluator.add_leaf(
        id="Initial_Report_Requirement",
        desc="Correctly identify the 3 calendar days (72 hours) initial report requirement",
        parent=tl_node,
        critical=True
    )
    await evaluator.verify(
        claim="An initial NORS report is due within 72 hours (3 calendar days) of the outage.",
        node=leaf_initial_report,
        sources=safe_urls((data.timeline or NORSTimeline()).timeline_reference_urls),
        additional_instruction="Verify the 72‑hour / 3‑day initial report requirement from the cited FCC/NORS source."
    )

    leaf_final_report = evaluator.add_leaf(
        id="Final_Report_Requirement",
        desc="Correctly identify the 30-day final report requirement",
        parent=tl_node,
        critical=True
    )
    await evaluator.verify(
        claim="A final NORS report is due within 30 days after the outage.",
        node=leaf_final_report,
        sources=safe_urls((data.timeline or NORSTimeline()).timeline_reference_urls),
        additional_instruction="Verify the 30‑day final report requirement from the cited FCC/NORS source."
    )

    leaf_timeline_ref = evaluator.add_leaf(
        id="Timeline_Reference",
        desc="Provide reference to FCC NORS timeline requirements",
        parent=tl_node,
        critical=True
    )
    await evaluator.verify(
        claim="The cited FCC sources explicitly state the 120‑minute notification, 72‑hour initial report, and 30‑day final report requirements.",
        node=leaf_timeline_ref,
        sources=safe_urls((data.timeline or NORSTimeline()).timeline_reference_urls),
        additional_instruction="Confirm the source contains the specific timeline values."
    )


async def verify_backup_power(evaluator: Evaluator, parent_node, data: OutageAnalysisExtraction) -> None:
    bp = data.backup or BackupPowerInfo()

    bp_root = evaluator.add_sequential(
        id="Backup_Power_Compliance_Assessment",
        desc="Evaluate whether the backup power systems met FCC requirements for cell sites",
        parent=parent_node,
        critical=False
    )

    # 1) FCC_Cell_Site_Standard_Identification (parallel, critical)
    std_node = evaluator.add_parallel(
        id="FCC_Cell_Site_Standard_Identification",
        desc="Identify the FCC backup power requirement for cell sites",
        parent=bp_root,
        critical=True
    )

    leaf_min_std = evaluator.add_leaf(
        id="Minimum_Backup_Duration_Standard",
        desc="Correctly identify that FCC requires cell sites to have minimum 8 hours of emergency backup power",
        parent=std_node,
        critical=True
    )
    await evaluator.verify(
        claim="FCC requires cell sites to have at least 8 hours of emergency backup power.",
        node=leaf_min_std,
        sources=safe_urls(bp.backup_requirement_reference_urls),
        additional_instruction="Verify the cited source states a minimum 8‑hour backup power requirement for cell sites."
    )

    leaf_std_ref = evaluator.add_leaf(
        id="Standard_Reference",
        desc="Provide reference to FCC backup power requirements",
        parent=std_node,
        critical=True
    )
    await evaluator.verify(
        claim="The cited sources are authoritative FCC backup power requirements for wireless cell sites.",
        node=leaf_std_ref,
        sources=safe_urls(bp.backup_requirement_reference_urls),
        additional_instruction="Confirm the source explicitly describes FCC backup power requirements (cell sites, minimum duration)."
    )

    # 2) Actual_Backup_Duration_Identification (critical leaf)
    leaf_actual = evaluator.add_leaf(
        id="Actual_Backup_Duration_Identification",
        desc="Correctly identify the actual backup power duration from the scenario",
        parent=bp_root,
        critical=True
    )
    await evaluator.verify(
        claim=f"Backup power at the affected sites lasted {ACTUAL_BACKUP_HOURS} hours.",
        node=leaf_actual,
        additional_instruction="Judge correct if the answer clearly states 5 hours of backup power at the affected sites."
    )

    # 3) Compliance_Determination (parallel, critical)
    comp_node = evaluator.add_parallel(
        id="Compliance_Determination",
        desc="Compare actual backup duration against FCC requirement and determine compliance status",
        parent=bp_root,
        critical=True
    )

    leaf_comp_analysis = evaluator.add_leaf(
        id="Compliance_Analysis",
        desc="Correctly determine whether the actual backup duration meets or fails to meet the FCC 8-hour minimum requirement",
        parent=comp_node,
        critical=True
    )
    await evaluator.verify(
        claim="Five hours of backup power does not meet an 8‑hour minimum requirement; therefore, the affected sites were not compliant.",
        node=leaf_comp_analysis,
        sources=safe_urls(bp.backup_requirement_reference_urls),
        additional_instruction="Use the cited 8‑hour minimum requirement to judge non‑compliance for 5 hours of backup power."
    )

    leaf_comp_ref = evaluator.add_leaf(
        id="Compliance_Reference",
        desc="Provide reference to FCC backup power compliance standards",
        parent=comp_node,
        critical=True
    )
    await evaluator.verify(
        claim="The cited sources define the FCC backup power compliance standard used for this determination.",
        node=leaf_comp_ref,
        sources=safe_urls(bp.compliance_reference_urls),
        additional_instruction="Confirm that the referenced standard is the basis for the compliance conclusion."
    )


async def verify_availability_sla(evaluator: Evaluator, parent_node, data: OutageAnalysisExtraction) -> None:
    av = data.availability or AvailabilityInfo()

    avail_root = evaluator.add_sequential(
        id="Network_Availability_Impact_Analysis",
        desc="Calculate the availability impact and evaluate SLA compliance",
        parent=parent_node,
        critical=False
    )

    # 1) Availability_Calculation (parallel, critical)
    calc_node = evaluator.add_parallel(
        id="Availability_Calculation",
        desc="Calculate the network availability percentage impact from this outage",
        parent=avail_root,
        critical=True
    )

    leaf_year_minutes = evaluator.add_leaf(
        id="Downtime_to_Annual_Conversion",
        desc="Correctly convert the outage duration to an annual downtime proportion",
        parent=calc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"A calendar year has {YEAR_MINUTES} minutes (365×24×60).",
        node=leaf_year_minutes,
        sources=safe_urls(av.availability_calc_reference_urls),
        additional_instruction="Verify or accept the standard figure 525,600 minutes per year."
    )

    leaf_av_pct = evaluator.add_leaf(
        id="Availability_Percentage_Calculation",
        desc="Correctly calculate the annual availability percentage if this were the only outage",
        parent=calc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"If this were the only outage, annual availability would be approximately {fmt_pct(CALCULATED_AVAILABILITY_PERCENT)}.",
        node=leaf_av_pct,
        sources=safe_urls(av.availability_calc_reference_urls),
        additional_instruction="Allow minor rounding differences (e.g., 99.931% vs 99.93%). Availability = 1 − (downtime minutes ÷ total minutes)."
    )

    leaf_calc_ref = evaluator.add_leaf(
        id="Calculation_Reference",
        desc="Provide reference to availability calculation methodology",
        parent=calc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Industry availability methodology computes availability as 1 − (downtime ÷ total time in the period).",
        node=leaf_calc_ref,
        sources=safe_urls(av.availability_calc_reference_urls),
        additional_instruction="Confirm the cited source explains the availability formula or equivalent calculation method."
    )

    # 2) SLA_Standards_Comparison (parallel, non-critical)
    sla_node = evaluator.add_parallel(
        id="SLA_Standards_Comparison",
        desc="Compare the calculated availability against the committed SLA standard",
        parent=avail_root,
        critical=False
    )

    leaf_sla_commit = evaluator.add_leaf(
        id="SLA_Commitment_Identification",
        desc="Correctly identify the provider's 99.99% SLA commitment from the scenario",
        parent=sla_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provider's SLA commitment is 99.99% availability.",
        node=leaf_sla_commit,
        additional_instruction="Judge correct if the answer clearly cites a 99.99% availability SLA."
    )

    leaf_allowable = evaluator.add_leaf(
        id="SLA_Allowable_Downtime",
        desc="Correctly identify the maximum allowable downtime for 99.99% availability standard",
        parent=sla_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"99.99% annual availability permits at most roughly {round(ALLOWABLE_DOWNTIME_99_99_MINUTES, 2)} minutes of downtime in a 365‑day year.",
        node=leaf_allowable,
        sources=safe_urls(av.standards_reference_urls),
        additional_instruction="Verify the downtime equivalent for 99.99% (≈ 52.56 minutes/year) from the cited standard or table."
    )

    leaf_violation = evaluator.add_leaf(
        id="SLA_Violation_Determination",
        desc="Correctly determine whether the outage duration violates the 99.99% SLA commitment",
        parent=sla_node,
        critical=True
    )
    await evaluator.verify(
        claim="A single 360‑minute outage would violate a 99.99% availability SLA.",
        node=leaf_violation,
        sources=safe_urls(av.standards_reference_urls),
        additional_instruction="Base this on 360 minutes > allowable ~52.56 minutes; confirm via cited availability standards if present."
    )

    leaf_std_ref = evaluator.add_leaf(
        id="Standards_Reference",
        desc="Provide reference to telecommunications availability standards definitions",
        parent=sla_node,
        critical=True
    )
    await evaluator.verify(
        claim="The cited standards define availability targets and their downtime equivalents (e.g., 99.99% ≈ 52 minutes/year).",
        node=leaf_std_ref,
        sources=safe_urls(av.standards_reference_urls),
        additional_instruction="Confirm the reference contains the mapping/definition of availability vs allowable downtime."
    )


async def verify_redundancy(evaluator: Evaluator, parent_node, data: OutageAnalysisExtraction) -> None:
    rd = data.redundancy or RedundancyInfo()

    red_root = evaluator.add_parallel(
        id="Network_Redundancy_Requirements",
        desc="Identify the redundancy and resilience standards that should have been in place",
        parent=parent_node,
        critical=False
    )

    # 1) N+1 Redundancy (parallel, critical)
    n1_node = evaluator.add_parallel(
        id="N_Plus_One_Redundancy_Standard",
        desc="Correctly identify that N+1 redundancy is the minimum standard for critical telecommunications equipment",
        parent=red_root,
        critical=True
    )

    leaf_n1_def = evaluator.add_leaf(
        id="N_Plus_One_Definition",
        desc="Correctly explain N+1 redundancy as having one backup component beyond minimum required capacity",
        parent=n1_node,
        critical=True
    )
    await evaluator.verify(
        claim="N+1 redundancy means one additional backup component beyond the number required (N) to meet capacity, so service continues if one component fails.",
        node=leaf_n1_def,
        sources=safe_urls(rd.n_plus_one_reference_urls),
        additional_instruction="Verify the cited source defines N+1 in these terms."
    )

    leaf_n1_ref = evaluator.add_leaf(
        id="N_Plus_One_Reference",
        desc="Provide reference to N+1 redundancy standards",
        parent=n1_node,
        critical=True
    )
    await evaluator.verify(
        claim="The cited sources define or standardize N+1 redundancy for critical infrastructure.",
        node=leaf_n1_ref,
        sources=safe_urls(rd.n_plus_one_reference_urls),
        additional_instruction="Confirm the reference explicitly discusses N+1 redundancy."
    )

    # 2) Geographic Diversity (parallel, critical)
    geo_node = evaluator.add_parallel(
        id="Geographic_Diversity_Standard",
        desc="Correctly identify that geographic diversity and diverse routing are essential for network resilience",
        parent=red_root,
        critical=True
    )

    leaf_geo_def = evaluator.add_leaf(
        id="Geographic_Diversity_Definition",
        desc="Correctly explain geographic diversity as physically separated network paths to prevent single-location failures",
        parent=geo_node,
        critical=True
    )
    await evaluator.verify(
        claim="Geographic diversity entails physically separated, diverse‑routed network paths/facilities to avoid single‑location failures.",
        node=leaf_geo_def,
        sources=safe_urls(rd.geographic_diversity_reference_urls),
        additional_instruction="Verify the cited source defines geographic diversity/diverse routing in these terms."
    )

    leaf_geo_ref = evaluator.add_leaf(
        id="Geographic_Diversity_Reference",
        desc="Provide reference to geographic diversity and diverse routing standards",
        parent=geo_node,
        critical=True
    )
    await evaluator.verify(
        claim="The cited sources are standards or authoritative references describing diverse routing/geographic diversity for telecom networks.",
        node=leaf_geo_ref,
        sources=safe_urls(rd.geographic_diversity_reference_urls),
        additional_instruction="Confirm the reference explicitly discusses geographic diversity or diverse routing."
    )

    # 3) Redundancy Purpose Explanation (non-critical custom existence)
    evaluator.add_custom_node(
        result=bool(rd.redundancy_purpose_text and rd.redundancy_purpose_text.strip()),
        id="Redundancy_Purpose_Explanation",
        desc="Explain how these redundancy standards help prevent or mitigate multi-site outages",
        parent=red_root,
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
    Evaluate an answer for the telecommunications outage compliance analysis task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates independent dimensions in parallel
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

    # Extract structured information (single comprehensive extraction)
    extraction = await evaluator.extract(
        prompt=prompt_extract_outage_analysis(),
        template_class=OutageAnalysisExtraction,
        extraction_name="outage_analysis_extraction",
    )

    # Add ground truth info and derived calculations for transparency
    evaluator.add_ground_truth({
        "scenario": {
            "cell_sites": SCENARIO_CELL_SITES,
            "users_per_site": SCENARIO_USERS_PER_SITE,
            "outage_duration_minutes": OUTAGE_DURATION_MINUTES,
            "backup_duration_hours": ACTUAL_BACKUP_HOURS,
            "sla_commitment_percent": "99.99%",
        },
        "fcc_thresholds": {
            "user_minutes_threshold": 900000,
            "duration_threshold_minutes": 30
        },
        "calculations": {
            "total_user_minutes": TOTAL_USER_MINUTES,
            "year_minutes": YEAR_MINUTES,
            "allowable_downtime_99_99_minutes": round(ALLOWABLE_DOWNTIME_99_99_MINUTES, 2),
            "calculated_availability_percent": round(CALCULATED_AVAILABILITY_PERCENT, 3)
        },
        "assumptions": {
            "backup_requirement_min_hours_expected": BACKUP_REQUIREMENT_MIN_HOURS
        }
    })

    # Build top-level parallel dimensions
    analysis_root = evaluator.add_parallel(
        id="Telecommunications_Outage_Compliance_Analysis",
        desc="Comprehensive evaluation of a telecommunications outage scenario against FCC regulations and industry standards across five independent dimensions",
        parent=root,
        critical=False  # Set non-critical for consistency with framework rules
    )

    # Run each dimension's verification
    await verify_nors_reporting(evaluator, analysis_root, extraction)
    await verify_backup_power(evaluator, analysis_root, extraction)
    await verify_availability_sla(evaluator, analysis_root, extraction)
    await verify_redundancy(evaluator, analysis_root, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()