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
TASK_ID = "psap_dr_plan_ca_tier2"
TASK_DESCRIPTION = """
Design a comprehensive telecommunications disaster recovery infrastructure plan for a 911 Public Safety Answering Point (PSAP) facility located in California's Tier 2 high fire threat district. Your plan must include: (1) Selection of two different nationwide telecommunications providers (one primary, one secondary) that both offer service in the California location, with the primary provider offering a minimum 99.99% uptime SLA and the secondary provider offering a minimum 99.9% uptime SLA; (2) Specification of backup power requirements including 72 hours of backup power for central office facilities (per California CPUC requirements for Tier 2/3 high fire threat districts) and 8 hours for cell sites (per FCC requirements); (3) Geographic diversity plan ensuring the primary and secondary providers use geographically diverse routing paths and physically separate network infrastructure; (4) Compliance plan addressing FCC 911 special facility notification requirements, including the capability to notify the 911 facility within 30 minutes of discovering an outage, using both telephone and electronic writing; (5) Compliance plan addressing FCC Network Outage Reporting System (NORS) requirements for outages lasting 30 minutes or longer, including initial notification within 120 minutes, Initial Communications Outage Report within 72 hours, and Final Communications Outage Report within 30 days; (6) Definition of a specific Recovery Time Objective (RTO) appropriate for 911 service criticality; (7) Network redundancy architecture specifying multiple independent network paths and automatic failover capability to the secondary provider upon primary failure; (8) Emergency power contingency plan including provisions for rapid generator fuel resupply during extended outages; (9) Real-time network monitoring system specification; (10) Documentation package including written SLA agreements from both providers and documentation demonstrating compliance with both FCC and California CPUC regulatory requirements. Provide specific provider names, concrete specifications for each requirement, and reference URLs documenting all regulatory requirements and provider capabilities.
"""

ALLOWED_PROVIDERS = ["AT&T", "Verizon", "T-Mobile"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProviderInfo(BaseModel):
    name: Optional[str] = None
    sla_uptime: Optional[str] = None  # e.g., "99.99%", "five nines"
    capability_urls: List[str] = Field(default_factory=list)  # SLA pages, coverage maps, enterprise service pages
    service_in_location_statement: Optional[bool] = None  # whether plan explicitly says provider serves the specified CA location


class PlanExtraction(BaseModel):
    # Identification of location
    ca_location: Optional[str] = None

    # Providers
    primary: ProviderInfo = Field(default_factory=ProviderInfo)
    secondary: ProviderInfo = Field(default_factory=ProviderInfo)

    # Backup power details + regulatory URLs
    backup_power_cpuc_central_office_hours: Optional[str] = None  # expect "72 hours" or equivalent
    backup_power_fcc_cell_site_hours: Optional[str] = None  # expect "8 hours" or equivalent
    backup_power_cpuc_urls: List[str] = Field(default_factory=list)
    backup_power_fcc_urls: List[str] = Field(default_factory=list)

    # Geographic/Physical diversity
    geographic_diverse_routing: Optional[bool] = None
    physically_separate_infrastructure: Optional[bool] = None

    # FCC 911 special facility notification (PSAP notifications)
    fcc_911_notify_within_30_min: Optional[bool] = None
    fcc_911_notification_modes_phone_and_electronic: Optional[bool] = None
    fcc_911_first_followup_within_2_hours: Optional[bool] = None
    fcc_911_reg_urls: List[str] = Field(default_factory=list)

    # FCC NORS reporting
    nors_applies_30_min_and_thresholds: Optional[bool] = None
    nors_notification_within_120_min: Optional[bool] = None
    nors_initial_report_within_72_hours: Optional[bool] = None
    nors_final_report_within_30_days: Optional[bool] = None
    nors_reg_urls: List[str] = Field(default_factory=list)

    # RTO
    rto_value: Optional[str] = None  # e.g., "15 minutes"
    rto_tied_to_911_criticality: Optional[bool] = None

    # Network redundancy
    multiple_independent_network_paths: Optional[bool] = None
    automatic_failover_to_secondary: Optional[bool] = None

    # Emergency power contingency
    rapid_generator_fuel_resupply: Optional[bool] = None

    # Monitoring
    monitoring_system_specified: Optional[bool] = None

    # Documentation package
    documentation_sla_agreements_included: Optional[bool] = None
    documentation_compliance_docs_included: Optional[bool] = None
    documentation_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    You must extract a structured summary of the PSAP telecommunications disaster recovery plan as presented in the answer.

    Extract the following fields exactly as described. If a field is not explicitly present, set it to null, and for URL arrays return an empty list.

    1) ca_location: The specific California city or location named for the PSAP. If not given, return null.

    2) primary:
       - name: Primary provider name as written (e.g., "AT&T", "Verizon", "T-Mobile"; accept variants like "AT&T Business").
       - sla_uptime: The stated uptime SLA text for the primary provider (e.g., "99.99%", "five nines"). If not clearly quantified, set null.
       - capability_urls: All URLs cited for the primary provider that document either SLA/availability guarantees or service availability/coverage (e.g., SLA pages, enterprise service docs, coverage maps).
       - service_in_location_statement: true if the plan text explicitly states the primary provider offers service in the specified California location; false if it explicitly states otherwise; null if not stated.

    3) secondary:
       - name: Secondary provider name as written.
       - sla_uptime: The stated uptime SLA text for the secondary provider (e.g., "99.9%").
       - capability_urls: All URLs cited for the secondary provider that document either SLA/availability guarantees or service availability/coverage.
       - service_in_location_statement: true/false/null as above.

    4) Backup power requirements and regulatory citations:
       - backup_power_cpuc_central_office_hours: The stated number of hours for central office backup power (expect "72 hours" if compliant). If not stated, null.
       - backup_power_fcc_cell_site_hours: The stated number of hours for cell site backup power (expect "8 hours" if compliant). If not stated, null.
       - backup_power_cpuc_urls: URLs that document the CPUC backup power requirements (e.g., decision orders, CPUC pages).
       - backup_power_fcc_urls: URLs that document the FCC backup power requirements for cell sites (e.g., 47 CFR § 9.20 or related FCC orders/fact sheets).

    5) Geographic/physical diversity:
       - geographic_diverse_routing: true if the plan explicitly calls for geographically diverse routing between primary and secondary; false if explicitly not; null if not stated.
       - physically_separate_infrastructure: true/false/null as above.

    6) FCC 911 special facility notifications:
       - fcc_911_notify_within_30_min: true if the plan states capability to notify PSAP within 30 minutes of discovering an outage; false/null otherwise.
       - fcc_911_notification_modes_phone_and_electronic: true if the plan states notification by both telephone and electronic writing; false/null otherwise.
       - fcc_911_first_followup_within_2_hours: true if the plan states first follow-up within 2 hours; false/null otherwise.
       - fcc_911_reg_urls: URLs that document the FCC 911 PSAP/special-facility outage notification requirements.

    7) FCC NORS reporting:
       - nors_applies_30_min_and_thresholds: true if the plan states NORS applies to outages lasting ≥ 30 minutes and meeting thresholds (e.g., ≥ 900,000 user-minutes or affecting 911 special facilities); false/null otherwise.
       - nors_notification_within_120_min: true if the plan states initial NORS notification within 120 minutes; false/null otherwise.
       - nors_initial_report_within_72_hours: true if the plan states Initial Report within 72 hours; false/null otherwise.
       - nors_final_report_within_30_days: true if the plan states Final Report within 30 days; false/null otherwise.
       - nors_reg_urls: URLs that document FCC NORS thresholds/timelines.

    8) RTO:
       - rto_value: The quantified Recovery Time Objective (e.g., "15 minutes"). If not quantified, null.
       - rto_tied_to_911_criticality: true if the plan explicitly ties RTO to 911 mission-critical needs; false/null otherwise.

    9) Network redundancy:
       - multiple_independent_network_paths: true if the plan specifies multiple independent network paths; false/null otherwise.
       - automatic_failover_to_secondary: true if the plan specifies automatic failover to secondary on primary failure; false/null otherwise.

    10) Emergency power contingency:
       - rapid_generator_fuel_resupply: true if the plan includes provisions for rapid generator fuel resupply; false/null otherwise.

    11) Monitoring:
       - monitoring_system_specified: true if the plan specifies a real-time network monitoring system/tool; false/null otherwise.

    12) Documentation package:
       - documentation_sla_agreements_included: true if the plan includes written SLA agreements from both providers; false/null otherwise.
       - documentation_compliance_docs_included: true if the plan includes documentation demonstrating compliance with FCC and CPUC requirements; false/null otherwise.
       - documentation_urls: any URLs included for SLA agreements or compliance docs (can overlap with the regulatory URL sets above).

    Return a single JSON object exactly matching the target schema field names.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_provider_selection_and_slas(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Provider_Selection_and_SLAs",
        desc="Select two different nationwide providers with required SLAs and California service availability, with supporting URLs",
        parent=parent,
        critical=True
    )

    # Primary provider eligibility
    if not plan.primary.name:
        evaluator.add_custom_node(
            result=False,
            id="Primary_Provider_Name_And_Eligibility",
            desc="Primary provider is explicitly named and is one of: AT&T, Verizon, T-Mobile",
            parent=node,
            critical=True
        )
    else:
        leaf = evaluator.add_leaf(
            id="Primary_Provider_Name_And_Eligibility",
            desc="Primary provider is explicitly named and is one of: AT&T, Verizon, T-Mobile",
            parent=node,
            critical=True
        )
        claim = f"The primary provider chosen is '{plan.primary.name}', and it is one of the following: AT&T, Verizon, or T-Mobile."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            additional_instruction="Treat variants like 'AT&T Business', 'Verizon Business', or 'T-Mobile for Business' as matches to their parent brands. Case-insensitive."
        )

    # Secondary provider eligibility and different from primary
    if not plan.secondary.name:
        evaluator.add_custom_node(
            result=False,
            id="Secondary_Provider_Name_And_Eligibility",
            desc="Secondary provider is explicitly named, is one of: AT&T, Verizon, T-Mobile, and is different from the primary provider",
            parent=node,
            critical=True
        )
    else:
        leaf = evaluator.add_leaf(
            id="Secondary_Provider_Name_And_Eligibility",
            desc="Secondary provider is explicitly named, is one of: AT&T, Verizon, T-Mobile, and is different from the primary provider",
            parent=node,
            critical=True
        )
        prim = plan.primary.name or "[unspecified]"
        claim = f"The secondary provider chosen is '{plan.secondary.name}', it is one of AT&T/Verizon/T-Mobile, and it is different from the primary provider '{prim}'."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            additional_instruction="Confirm both eligibility (among the three) and that secondary is not the same as primary. Case-insensitive and allow brand variants."
        )

    # Both providers service availability in CA location (statement in plan)
    leaf = evaluator.add_leaf(
        id="Both_Providers_Service_Availability_CA_Location",
        desc="Plan states that both providers offer service in the specified California location",
        parent=node,
        critical=True
    )
    loc = plan.ca_location or "the specified California location"
    p_name = plan.primary.name or "the primary provider"
    s_name = plan.secondary.name or "the secondary provider"
    claim = f"The plan explicitly states that both {p_name} and {s_name} offer service in {loc}, California."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction="Look for explicit service availability statements tied to the named California location. Generic nationwide availability alone is insufficient."
    )

    # Primary SLA ≥ 99.99%
    leaf = evaluator.add_leaf(
        id="Primary_SLA_Meets_99_99",
        desc="Primary provider SLA is stated as ≥ 99.99% uptime",
        parent=node,
        critical=True
    )
    prim_sla = plan.primary.sla_uptime or "[unspecified]"
    await evaluator.verify(
        claim=f"The plan states the primary provider's uptime SLA is at least 99.99% (the extracted value is '{prim_sla}').",
        node=leaf,
        additional_instruction="Treat 'four nines', '99.99%', or higher (e.g., 99.995%, 99.999%) as meeting the requirement. If no explicit figure is stated, this should be judged incorrect."
    )

    # Secondary SLA ≥ 99.9%
    leaf = evaluator.add_leaf(
        id="Secondary_SLA_Meets_99_9",
        desc="Secondary provider SLA is stated as ≥ 99.9% uptime",
        parent=node,
        critical=True
    )
    sec_sla = plan.secondary.sla_uptime or "[unspecified]"
    await evaluator.verify(
        claim=f"The plan states the secondary provider's uptime SLA is at least 99.9% (the extracted value is '{sec_sla}').",
        node=leaf,
        additional_instruction="Treat 'three nines', '99.9%', or higher as meeting the requirement. If no explicit figure is stated, this should be judged incorrect."
    )

    # Provider capabilities URLs (expand into two leaves to ensure both providers have evidence)
    caps_group = evaluator.add_parallel(
        id="Provider_Capabilities_URLs",
        desc="Plan provides reference URL(s) documenting provider capabilities (e.g., SLA and/or service availability) for both providers",
        parent=node,
        critical=True
    )
    # Primary URLs
    if not plan.primary.capability_urls:
        evaluator.add_custom_node(
            result=False,
            id="Provider_Capabilities_URLs_Primary",
            desc="Primary provider capability URL(s) present and support SLA/coverage",
            parent=caps_group,
            critical=True
        )
    else:
        leaf = evaluator.add_leaf(
            id="Provider_Capabilities_URLs_Primary",
            desc="Primary provider capability URL(s) present and support SLA/coverage",
            parent=caps_group,
            critical=True
        )
        await evaluator.verify(
            claim=f"At least one of these webpages documents {p_name}'s enterprise/service capabilities relevant to SLAs and/or service availability/coverage (preferably in California).",
            node=leaf,
            sources=plan.primary.capability_urls,
            additional_instruction="Accept official provider product/SLA pages, enterprise service catalogs, or official coverage maps/availability checkers. Third-party credible docs are acceptable if clearly authoritative."
        )
    # Secondary URLs
    if not plan.secondary.capability_urls:
        evaluator.add_custom_node(
            result=False,
            id="Provider_Capabilities_URLs_Secondary",
            desc="Secondary provider capability URL(s) present and support SLA/coverage",
            parent=caps_group,
            critical=True
        )
    else:
        leaf = evaluator.add_leaf(
            id="Provider_Capabilities_URLs_Secondary",
            desc="Secondary provider capability URL(s) present and support SLA/coverage",
            parent=caps_group,
            critical=True
        )
        await evaluator.verify(
            claim=f"At least one of these webpages documents {s_name}'s enterprise/service capabilities relevant to SLAs and/or service availability/coverage (preferably in California).",
            node=leaf,
            sources=plan.secondary.capability_urls,
            additional_instruction="Accept official provider product/SLA pages, enterprise service catalogs, or official coverage maps/availability checkers. Third-party credible docs are acceptable if clearly authoritative."
        )


async def verify_backup_power_requirements(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Backup_Power_Requirements",
        desc="Backup power requirements for central offices and cell sites with regulatory URLs",
        parent=parent,
        critical=True
    )

    # CPUC 72 hours (central office)
    leaf = evaluator.add_leaf(
        id="Central_Office_72_Hours_CPUC",
        desc="Plan specifies 72 hours of backup power for central office facilities for Tier 2/3 high fire threat districts (CPUC requirement)",
        parent=node,
        critical=True
    )
    hours_text = plan.backup_power_cpuc_central_office_hours or "[unspecified]"
    await evaluator.verify(
        claim=f"The plan explicitly specifies 72 hours of backup power for central office facilities serving the PSAP in Tier 2/3 High Fire-Threat Districts (extracted text: '{hours_text}').",
        node=leaf,
        additional_instruction="The statement should clearly indicate 72 hours at central office or equivalent hub sites per CPUC. If a different duration is stated or not present, judge incorrect."
    )

    # FCC 8 hours (cell sites)
    leaf = evaluator.add_leaf(
        id="Cell_Sites_8_Hours_FCC",
        desc="Plan specifies minimum 8 hours of backup power for cell sites (FCC 47 CFR § 9.20 requirement)",
        parent=node,
        critical=True
    )
    cell_text = plan.backup_power_fcc_cell_site_hours or "[unspecified]"
    await evaluator.verify(
        claim=f"The plan explicitly specifies a minimum of 8 hours of backup power for cell sites (extracted text: '{cell_text}').",
        node=leaf,
        additional_instruction="Look for '8 hours' minimum at wireless cell sites as a regulatory baseline. If different or missing, judge incorrect."
    )

    # Regulatory URLs group (split into CPUC and FCC leaves)
    regs = evaluator.add_parallel(
        id="Backup_Power_Regulatory_URLs",
        desc="Plan provides reference URL(s) for CPUC and FCC backup power requirements",
        parent=node,
        critical=True
    )
    # CPUC URLs
    if not plan.backup_power_cpuc_urls:
        evaluator.add_custom_node(
            result=False,
            id="Backup_Power_Regulatory_URLs_CPUC",
            desc="CPUC backup power requirement URL(s) provided and supportive",
            parent=regs,
            critical=True
        )
    else:
        leaf = evaluator.add_leaf(
            id="Backup_Power_Regulatory_URLs_CPUC",
            desc="CPUC backup power requirement URL(s) provided and supportive",
            parent=regs,
            critical=True
        )
        await evaluator.verify(
            claim="This webpage documents CPUC requirements mandating at least 72 hours of backup power at central offices or equivalent network hubs serving communities in Tier 2/3 High Fire-Threat Districts.",
            node=leaf,
            sources=plan.backup_power_cpuc_urls,
            additional_instruction="Look for CPUC decisions, rulemakings, or official CPUC pages referencing 72 hours in HFTD contexts."
        )
    # FCC URLs
    if not plan.backup_power_fcc_urls:
        evaluator.add_custom_node(
            result=False,
            id="Backup_Power_Regulatory_URLs_FCC",
            desc="FCC backup power requirement URL(s) provided and supportive",
            parent=regs,
            critical=True
        )
    else:
        leaf = evaluator.add_leaf(
            id="Backup_Power_Regulatory_URLs_FCC",
            desc="FCC backup power requirement URL(s) provided and supportive",
            parent=regs,
            critical=True
        )
        await evaluator.verify(
            claim="This webpage documents FCC requirements indicating a minimum of 8 hours of backup power at cell sites (e.g., 47 CFR § 9.20 or related FCC orders/guidance).",
            node=leaf,
            sources=plan.backup_power_fcc_urls,
            additional_instruction="Accept FCC rules, orders, or official guidance that clearly specify the 8-hour minimum for wireless facilities."
        )


async def verify_geodiversity(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Geographic_and_Physical_Diversity",
        desc="Geographic diversity and physically separate infrastructure between primary and secondary providers",
        parent=parent,
        critical=True
    )

    # Geographically diverse routing
    leaf = evaluator.add_leaf(
        id="Geographically_Diverse_Routing",
        desc="Plan specifies geographically diverse routing paths between primary and secondary providers",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies that the primary and secondary providers will use geographically diverse routing paths.",
        node=leaf,
        additional_instruction="Look for statements about diverse fiber routes, separate entry paths, different long-haul paths, or similar."
    )

    # Physically separate infrastructure
    leaf = evaluator.add_leaf(
        id="Physically_Separate_Infrastructure",
        desc="Plan specifies physically separate network infrastructure between primary and secondary providers",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies physically separate network infrastructure between the two providers (e.g., separate conduits, demarcs, racks, or facilities).",
        node=leaf,
        additional_instruction="Any clear statement committing to physical separation counts."
    )


async def verify_fcc_911_notification(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="FCC_911_Special_Facility_Notification",
        desc="FCC 911 special facility notification compliance elements with regulatory URL",
        parent=parent,
        critical=True
    )

    # 30-minute initial notification capability
    leaf = evaluator.add_leaf(
        id="Notify_Within_30_Minutes",
        desc="Plan states capability to notify the 911 facility within 30 minutes of discovering an outage",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan states capability to notify the 911 PSAP within 30 minutes of discovering an outage.",
        node=leaf
    )

    # Modes: telephone and electronic writing
    leaf = evaluator.add_leaf(
        id="Notification_Modes_Telephone_And_Electronic_Writing",
        desc="Plan states notification will be via both telephone and electronic writing",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan states that outage notifications to the PSAP will be made by both telephone and electronic writing.",
        node=leaf
    )

    # First follow-up within 2 hours
    leaf = evaluator.add_leaf(
        id="First_Followup_Within_2_Hours",
        desc="Plan states first follow-up notification within 2 hours after initial contact",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan states that the first follow-up notification will occur within 2 hours after the initial contact to the PSAP.",
        node=leaf
    )

    # Regulatory URL(s) supporting the requirement
    if not plan.fcc_911_reg_urls:
        evaluator.add_custom_node(
            result=False,
            id="FCC_911_Notification_Regulatory_URL",
            desc="Plan provides reference URL(s) documenting the FCC 911 special facility notification requirements",
            parent=node,
            critical=True
        )
    else:
        leaf = evaluator.add_leaf(
            id="FCC_911_Notification_Regulatory_URL",
            desc="Plan provides reference URL(s) documenting the FCC 911 special facility notification requirements",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim="This webpage documents the FCC's 911 PSAP outage notification requirements (including initial notifications within 30 minutes and follow-up updates, by phone and written/electronic means).",
            node=leaf,
            sources=plan.fcc_911_reg_urls,
            additional_instruction="Accept FCC rules, orders, or official notices/fact sheets that clearly articulate PSAP outage notification obligations and timelines."
        )


async def verify_nors(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="FCC_NORS_Reporting",
        desc="FCC NORS reporting compliance elements with regulatory URL",
        parent=parent,
        critical=True
    )

    # Applies to outages ≥ 30 minutes and meeting thresholds
    leaf = evaluator.add_leaf(
        id="NORS_Applies_To_Outages_30_Min_And_Thresholds",
        desc="Plan states NORS reporting applies to outages lasting ≥ 30 minutes and meeting stated thresholds (including ≥ 900,000 user-minutes or affecting 911 special facilities)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan states that NORS reporting applies to outages lasting at least 30 minutes that meet specified thresholds (e.g., ≥ 900,000 user-minutes or affecting 911 special facilities).",
        node=leaf
    )

    # Initial notification within 120 minutes
    leaf = evaluator.add_leaf(
        id="NORS_Notification_Within_120_Min",
        desc="Plan states initial NORS notification within 120 minutes of outage discovery",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan states that the initial NORS notification will be filed within 120 minutes of outage discovery.",
        node=leaf
    )

    # Initial report within 72 hours
    leaf = evaluator.add_leaf(
        id="NORS_Initial_Report_Within_72_Hours",
        desc="Plan states Initial Communications Outage Report within 72 hours",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan states that the Initial Communications Outage Report will be submitted within 72 hours.",
        node=leaf
    )

    # Final report within 30 days
    leaf = evaluator.add_leaf(
        id="NORS_Final_Report_Within_30_Days",
        desc="Plan states Final Communications Outage Report within 30 days",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan states that the Final Communications Outage Report will be submitted within 30 days.",
        node=leaf
    )

    # Regulatory URL(s) supporting NORS requirements
    if not plan.nors_reg_urls:
        evaluator.add_custom_node(
            result=False,
            id="NORS_Regulatory_URL",
            desc="Plan provides reference URL(s) documenting FCC NORS requirements",
            parent=node,
            critical=True
        )
    else:
        leaf = evaluator.add_leaf(
            id="NORS_Regulatory_URL",
            desc="Plan provides reference URL(s) documenting FCC NORS requirements",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim="This webpage documents the FCC's NORS reporting requirements, including applicable thresholds and timelines (initial notice within 120 minutes, initial report within 72 hours, final report within 30 days).",
            node=leaf,
            sources=plan.nors_reg_urls,
            additional_instruction="Accept FCC rules/orders/fact sheets that clearly specify NORS thresholds and reporting timelines."
        )


async def verify_rto(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    # RTO Definition (single critical leaf)
    leaf = evaluator.add_leaf(
        id="RTO_Definition",
        desc="Plan defines a Recovery Time Objective (RTO) with a specific quantified value and ties it to 911 service criticality (i.e., makes clear it is chosen for 911 criticality/mission-critical needs)",
        parent=parent,
        critical=True
    )
    rto_val = plan.rto_value or "[unspecified]"
    tie = "explicitly tied to 911 mission-critical needs" if plan.rto_tied_to_911_criticality else "tied to 911 mission-critical needs"
    await evaluator.verify(
        claim=f"The plan defines a quantified Recovery Time Objective of {rto_val} and it is {tie}.",
        node=leaf,
        additional_instruction="A numeric or clearly quantified value must be present (e.g., '15 minutes'). The narrative should connect the RTO to 911 mission-critical service needs."
    )


async def verify_network_redundancy(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Network_Redundancy_Architecture",
        desc="Network redundancy design includes independent paths and automatic failover",
        parent=parent,
        critical=True
    )

    # Multiple independent network paths
    leaf = evaluator.add_leaf(
        id="Multiple_Independent_Network_Paths",
        desc="Plan specifies multiple independent network paths",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies multiple independent network paths.",
        node=leaf,
        additional_instruction="Look for language like 'dual path', 'diverse circuits', 'independent routes', etc."
    )

    # Automatic failover to secondary
    leaf = evaluator.add_leaf(
        id="Automatic_Failover_To_Secondary",
        desc="Plan specifies automatic failover to the secondary provider upon primary failure",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies automatic failover to the secondary provider when the primary provider fails.",
        node=leaf,
        additional_instruction="Any automatic switchover/failover mechanism that moves traffic to the secondary provider qualifies."
    )


async def verify_emergency_power_contingency(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Emergency_Power_Contingency_Fuel_Resupply",
        desc="Emergency power contingency planning includes rapid generator fuel resupply",
        parent=parent,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="Rapid_Generator_Fuel_Resupply",
        desc="Plan includes provisions for rapid generator fuel resupply during extended outages",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes provisions for rapid generator fuel resupply during extended outages.",
        node=leaf,
        additional_instruction="Look for fuel vendor contracts/MOUs, on-call delivery procedures, or similar concrete resupply steps."
    )


async def verify_monitoring(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Real_Time_Network_Monitoring",
        desc="Real-time network monitoring system is specified",
        parent=parent,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="Monitoring_System_Specified",
        desc="Plan specifies a real-time network monitoring system",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies a real-time network monitoring system or tool for the telecommunications infrastructure.",
        node=leaf,
        additional_instruction="Named tools (e.g., SolarWinds, PRTG, Datadog, Splunk, ThousandEyes) or equivalent monitoring/NMS description is acceptable."
    )


async def verify_documentation_package(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Documentation_Package",
        desc="Required documentation package (SLAs and regulatory compliance documentation)",
        parent=parent,
        critical=True
    )

    # Written SLA agreements included
    leaf = evaluator.add_leaf(
        id="Written_SLA_Agreements_Included",
        desc="Plan includes written SLA agreements from both providers",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes written SLA agreements from both the primary and secondary providers.",
        node=leaf,
        additional_instruction="Look for explicit statements that written SLAs from both providers are included or attached."
    )

    # Compliance documentation included (FCC & CPUC)
    leaf = evaluator.add_leaf(
        id="Compliance_Documentation_FCC_And_CPUC",
        desc="Plan includes documentation demonstrating compliance with FCC and California CPUC regulatory requirements",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes documentation demonstrating compliance with both FCC and California CPUC regulatory requirements.",
        node=leaf,
        additional_instruction="Look for explicit statements about including compliance documentation; it can reference URLs, appendices, or attachments."
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
    Evaluate an answer for the PSAP telecommunications disaster recovery plan in CA Tier 2 HFTD.
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

    # Create top-level critical solution node
    solution_node = evaluator.add_parallel(
        id="Telecommunications_Infrastructure_Solution",
        desc="Telecommunications disaster recovery infrastructure plan for a 911 PSAP facility in California's Tier 2 high fire threat district, satisfying all stated requirements and constraints",
        parent=root,
        critical=True
    )

    # Extract structured information
    plan = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction"
    )

    # Add ground truths / expectations for transparency
    evaluator.add_ground_truth({
        "allowed_providers": ALLOWED_PROVIDERS,
        "primary_min_sla": ">= 99.99%",
        "secondary_min_sla": ">= 99.9%",
        "backup_power_requirements": {
            "CPUC_central_office": "72 hours (Tier 2/3 HFTD)",
            "FCC_cell_sites": "8 hours"
        },
        "fcc_911_notifications": {
            "initial": "within 30 minutes (phone + electronic writing)",
            "first_followup": "within 2 hours"
        },
        "nors_reporting": {
            "applies": ">= 30 minutes + thresholds (e.g., >= 900,000 user-minutes or affecting 911)",
            "initial_notice": "within 120 minutes",
            "initial_report": "within 72 hours",
            "final_report": "within 30 days"
        },
        "rto": "quantified value appropriate to 911 mission-critical service",
        "redundancy": "independent paths + automatic failover",
        "emergency_power": "rapid generator fuel resupply",
        "monitoring": "real-time network monitoring",
        "documentation": "written SLAs (both providers) + FCC/CPUC compliance docs"
    })

    # Build verification tree according to rubric
    await verify_provider_selection_and_slas(evaluator, solution_node, plan)
    await verify_backup_power_requirements(evaluator, solution_node, plan)
    await verify_geodiversity(evaluator, solution_node, plan)
    await verify_fcc_911_notification(evaluator, solution_node, plan)
    await verify_nors(evaluator, solution_node, plan)
    await verify_rto(evaluator, solution_node, plan)
    await verify_network_redundancy(evaluator, solution_node, plan)
    await verify_emergency_power_contingency(evaluator, solution_node, plan)
    await verify_monitoring(evaluator, solution_node, plan)
    await verify_documentation_package(evaluator, solution_node, plan)

    # Return evaluator summary
    return evaluator.get_summary()