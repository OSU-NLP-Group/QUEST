import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "telecom_florida_fcc_compliance"
TASK_DESCRIPTION = """
Identify a major telecommunications provider that operates cellular networks in Florida and verify their compliance with current FCC emergency preparedness and network reliability requirements. Specifically, you must confirm: (1) The provider's name and evidence of their cellular network operations in Florida; (2) Their compliance with the FCC backup power requirement (minimum 8 hours of backup power at cell sites); (3) Their procedures for submitting initial Network Outage Reporting System (NORS) reports within 72 hours of discovering covered outages; (4) Their procedures for submitting final NORS reports within 30 days of discovering outages; (5) Their compliance with PSAP notification requirements (notifying Public Safety Answering Points within 30 minutes of discovering 911-impacting outages, effective April 15, 2025); (6) Their procedures for providing follow-up PSAP notifications every 2 hours during ongoing 911-impacting outages. Additionally, if available, provide information about: the backup power technology they use (generators, batteries, fuel cells, or combination) and their tower inspection schedule and preventive maintenance program. For each compliance requirement, you must provide supporting URL evidence from reliable sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProviderComplianceExtraction(BaseModel):
    # Provider identification and Florida operations
    provider_name: Optional[str] = None
    major_provider_urls: List[str] = Field(default_factory=list)
    florida_ops_urls: List[str] = Field(default_factory=list)

    # Backup power: minimum 8 hours at cell sites
    backup_power_8h_statement: Optional[str] = None
    backup_power_8h_urls: List[str] = Field(default_factory=list)

    # NORS initial within 72 hours
    nors_initial_72h_statement: Optional[str] = None
    nors_initial_72h_urls: List[str] = Field(default_factory=list)

    # NORS final within 30 days
    nors_final_30d_statement: Optional[str] = None
    nors_final_30d_urls: List[str] = Field(default_factory=list)

    # PSAP 30-minute notification (effective 2025-04-15)
    psap_30m_effective_2025_statement: Optional[str] = None
    psap_30m_effective_2025_urls: List[str] = Field(default_factory=list)

    # PSAP follow-up every 2 hours during ongoing 911-impacting outages
    psap_2h_followup_statement: Optional[str] = None
    psap_2h_followup_urls: List[str] = Field(default_factory=list)

    # Tower inspection frequencies
    guyed_3y_statement: Optional[str] = None
    guyed_3y_urls: List[str] = Field(default_factory=list)

    selfsupport_5y_statement: Optional[str] = None
    selfsupport_5y_urls: List[str] = Field(default_factory=list)

    # Optional: backup power technology used
    backup_power_tech_statement: Optional[str] = None
    backup_power_tech_urls: List[str] = Field(default_factory=list)

    # Optional: preventive maintenance program / inspection schedule
    preventive_maintenance_statement: Optional[str] = None
    preventive_maintenance_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_provider_compliance() -> str:
    return """
    Extract structured information from the answer about a telecommunications provider's Florida operations and compliance with FCC/NORS/PSAP requirements. Return exactly the following fields:

    1) provider_name: The provider/carrier name explicitly mentioned (single provider; if multiple are mentioned, pick the primary one the answer evaluates).
    2) major_provider_urls: All URLs cited that support that the provider is a major/national U.S. telecommunications provider (e.g., nationwide carrier/MNO).
    3) florida_ops_urls: All URLs cited that directly support that the provider operates cellular (mobile wireless) networks in Florida (e.g., official coverage pages, Florida service pages, reliable news releases, filings referencing Florida operations).

    For each of the following compliance items, extract:
    - a *_statement string containing the exact sentence or concise paraphrase (from the answer) that the answer uses to assert the compliance/procedure; return null if the answer does not explicitly state it.
    - a corresponding *_urls array containing all URLs the answer cites to support that item; return an empty list if none are present.

    4) backup_power_8h_statement; backup_power_8h_urls
       The answer’s explicit statement that the provider maintains at least 8 hours (or "minimum 8 hours") of backup power at cell sites.
    5) nors_initial_72h_statement; nors_initial_72h_urls
       The answer’s explicit statement that the provider submits an initial NORS report within 72 hours (3 calendar days) of discovering a covered outage.
    6) nors_final_30d_statement; nors_final_30d_urls
       The answer’s explicit statement that the provider submits a final NORS report within 30 days of discovery of a covered outage.
    7) psap_30m_effective_2025_statement; psap_30m_effective_2025_urls
       The answer’s explicit statement that the provider notifies affected PSAPs within 30 minutes of discovering a potential 911-impacting outage, effective April 15, 2025.
    8) psap_2h_followup_statement; psap_2h_followup_urls
       The answer’s explicit statement that the provider provides follow-up PSAP notifications every 2 hours during ongoing 911-impacting outages.

    9) guyed_3y_statement; guyed_3y_urls
       The answer’s explicit statement that guyed towers are inspected at least once every 3 years.
    10) selfsupport_5y_statement; selfsupport_5y_urls
       The answer’s explicit statement that self-supporting towers are inspected at least once every 5 years.

    Optional items (include only if the answer provides them):
    11) backup_power_tech_statement; backup_power_tech_urls
        The answer’s description of backup power technology used (generators, batteries, fuel cells, or combination) and all supporting URLs.
    12) preventive_maintenance_statement; preventive_maintenance_urls
        The answer’s high-level description of tower inspection schedule and/or preventive maintenance program and all supporting URLs.

    Special rules for URL extraction:
    - Extract only URLs explicitly present in the answer (including markdown links).
    - If the answer provides no URL for a field, return an empty list for that field.
    - Do not fabricate URLs.

    Return a single JSON object conforming exactly to the specified schema fields.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_text(s: Optional[str]) -> bool:
    return bool(s is not None and str(s).strip())


def has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification sections                                                       #
# --------------------------------------------------------------------------- #
async def build_provider_identification_section(
    evaluator: Evaluator,
    parent,
    ext: ProviderComplianceExtraction,
) -> None:
    """
    Build and verify: Provider identification and Florida operations.
    """
    node = evaluator.add_parallel(
        id="ProviderIdentificationAndFloridaOperations",
        desc="Identify the provider and substantiate that it is a major provider and operates cellular networks in Florida.",
        parent=parent,
        critical=True,
    )

    # Provider name presence (critical)
    evaluator.add_custom_node(
        result=has_text(ext.provider_name),
        id="ProviderNameProvided",
        desc="A telecommunications provider name is provided.",
        parent=node,
        critical=True,
    )

    provider_name = ext.provider_name or "the provider"

    # Major provider evidence - require URLs present as a gating step (critical)
    evaluator.add_custom_node(
        result=has_urls(ext.major_provider_urls),
        id="ProviderIsMajorProviderEvidenceProvided",
        desc="At least one URL is provided to support that the provider is a major U.S. telecommunications provider.",
        parent=node,
        critical=True,
    )

    # Verify major provider status supported by URLs (critical)
    major_leaf = evaluator.add_leaf(
        id="ProviderIsMajorProvider",
        desc="The answer includes an objective justification that the provider is a major telecommunications provider (e.g., widely recognized major/national carrier), supported by at least one reliable URL.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{provider_name} is a major U.S. telecommunications provider (e.g., a nationwide cellular carrier).",
        node=major_leaf,
        sources=ext.major_provider_urls,
        additional_instruction=(
            "Accept evidence such as official corporate descriptions, market share/coverage summaries, FCC/NORS-related filings, or reputable analyses "
            "that indicate the provider is a major/national carrier/MNO. The URL(s) must directly support the claim."
        ),
    )

    # Florida operations evidence - URLs present gating (critical)
    evaluator.add_custom_node(
        result=has_urls(ext.florida_ops_urls),
        id="FloridaCellularOperationsEvidenceURLsProvided",
        desc="At least one reliable URL is provided that directly supports that the provider operates cellular networks in Florida.",
        parent=node,
        critical=True,
    )

    # Verify Florida operations supported by URLs (critical)
    fl_leaf = evaluator.add_leaf(
        id="FloridaCellularOperationsEvidence",
        desc="At least one reliable URL is provided that directly supports that the provider operates cellular networks in Florida.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{provider_name} operates cellular (mobile wireless) networks in the state of Florida.",
        node=fl_leaf,
        sources=ext.florida_ops_urls,
        additional_instruction=(
            "Accept evidence such as official coverage maps including Florida, Florida-specific service pages, official filings, or reliable news/press releases "
            "that clearly indicate active cellular network operations in Florida."
        ),
    )


async def build_requirements_section(
    evaluator: Evaluator,
    parent,
    ext: ProviderComplianceExtraction,
) -> None:
    """
    Build and verify all FCC/NORS/PSAP and reliability requirements (critical).
    """
    req = evaluator.add_parallel(
        id="FCC_NORS_PSAP_and_ReliabilityRequirements",
        desc="For each specified requirement, the answer states the required compliance/procedure and provides supporting reliable URL evidence.",
        parent=parent,
        critical=True,
    )

    # 1) Backup power: minimum 8 hours at cell sites
    bp = evaluator.add_parallel(
        id="BackupPowerMinimum8Hours",
        desc="Backup power: minimum 8 hours at cell sites.",
        parent=req,
        critical=True,
    )
    # Claim presence (critical)
    evaluator.add_custom_node(
        result=has_text(ext.backup_power_8h_statement),
        id="BackupPower8HoursClaim",
        desc="The answer explicitly states the provider maintains at least 8 hours of backup power at cell sites.",
        parent=bp,
        critical=True,
    )
    # Evidence URLs present (critical)
    evaluator.add_custom_node(
        result=has_urls(ext.backup_power_8h_urls),
        id="BackupPower8HoursEvidenceURLsProvided",
        desc="At least one reliable URL is provided that directly supports the 8-hour backup power claim.",
        parent=bp,
        critical=True,
    )
    # Evidence verification (critical)
    bp_ev = evaluator.add_leaf(
        id="BackupPower8HoursEvidenceURL",
        desc="At least one reliable URL is provided that directly supports the 8-hour backup power claim.",
        parent=bp,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provider maintains at least 8 hours of backup power at its cell sites.",
        node=bp_ev,
        sources=ext.backup_power_8h_urls,
        additional_instruction=(
            "The supporting page(s) should clearly state a minimum or at-least 8 hours of backup power (at cell sites, macro sites, or equivalent). "
            "Phrasings like 'at least 8 hours' or 'minimum 8 hours' are acceptable."
        ),
    )

    # 2) NORS initial within 72 hours of discovery
    nors_init = evaluator.add_parallel(
        id="NORSInitialReportWithin72Hours",
        desc="Initial NORS reporting: within 72 hours (3 calendar days) of discovering covered outages.",
        parent=req,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_text(ext.nors_initial_72h_statement),
        id="InitialNORS72HoursProcedureClaim",
        desc="The answer explicitly states the provider’s procedure/compliance is to submit an initial NORS report within 72 hours of discovery of a covered outage.",
        parent=nors_init,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_urls(ext.nors_initial_72h_urls),
        id="InitialNORS72HoursEvidenceURLsProvided",
        desc="At least one reliable URL is provided that directly supports the 72-hour initial NORS reporting procedure/timeline claim.",
        parent=nors_init,
        critical=True,
    )
    nors_init_ev = evaluator.add_leaf(
        id="InitialNORS72HoursEvidenceURL",
        desc="At least one reliable URL is provided that directly supports the 72-hour initial NORS reporting procedure/timeline claim.",
        parent=nors_init,
        critical=True,
    )
    await evaluator.verify(
        claim="The provider submits an initial NORS report within 72 hours (3 calendar days) of discovering a covered outage.",
        node=nors_init_ev,
        sources=ext.nors_initial_72h_urls,
        additional_instruction=(
            "The page(s) should explicitly mention an initial NORS reporting deadline of within 72 hours (or three calendar days) of discovery."
        ),
    )

    # 3) NORS final within 30 days of discovery
    nors_final = evaluator.add_parallel(
        id="NORSFinalReportWithin30Days",
        desc="Final NORS reporting: within 30 days of discovering outages.",
        parent=req,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_text(ext.nors_final_30d_statement),
        id="FinalNORS30DaysProcedureClaim",
        desc="The answer explicitly states the provider’s procedure/compliance is to submit a final NORS report within 30 days of discovery of a covered outage.",
        parent=nors_final,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_urls(ext.nors_final_30d_urls),
        id="FinalNORS30DaysEvidenceURLsProvided",
        desc="At least one reliable URL is provided that directly supports the 30-day final NORS reporting procedure/timeline claim.",
        parent=nors_final,
        critical=True,
    )
    nors_final_ev = evaluator.add_leaf(
        id="FinalNORS30DaysEvidenceURL",
        desc="At least one reliable URL is provided that directly supports the 30-day final NORS reporting procedure/timeline claim.",
        parent=nors_final,
        critical=True,
    )
    await evaluator.verify(
        claim="The provider submits a final NORS report within 30 days of discovery of the outage.",
        node=nors_final_ev,
        sources=ext.nors_final_30d_urls,
        additional_instruction=(
            "The supporting page(s) should explicitly reference a final NORS reporting deadline within 30 days of discovery."
        ),
    )

    # 4) PSAP notify within 30 minutes (effective 2025-04-15)
    psap_30 = evaluator.add_parallel(
        id="PSAPNotifyWithin30MinutesEffective2025_04_15",
        desc="PSAP notification: within 30 minutes for 911-impacting outages (effective April 15, 2025).",
        parent=req,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_text(ext.psap_30m_effective_2025_statement),
        id="PSAP30MinuteProcedureClaim",
        desc="The answer explicitly states the provider’s procedure/compliance is to notify affected PSAPs within 30 minutes of discovering a potential 911-impacting outage, effective April 15, 2025.",
        parent=psap_30,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_urls(ext.psap_30m_effective_2025_urls),
        id="PSAP30MinuteEvidenceURLsProvided",
        desc="At least one reliable URL is provided that directly supports the PSAP 30-minute notification procedure/requirement claim (including the effective date).",
        parent=psap_30,
        critical=True,
    )
    psap_30_ev = evaluator.add_leaf(
        id="PSAP30MinuteEvidenceURL",
        desc="At least one reliable URL is provided that directly supports the PSAP 30-minute notification procedure/requirement claim (including the effective date).",
        parent=psap_30,
        critical=True,
    )
    await evaluator.verify(
        claim="The provider notifies affected PSAPs within 30 minutes of discovering a potential 911-impacting outage, effective April 15, 2025.",
        node=psap_30_ev,
        sources=ext.psap_30m_effective_2025_urls,
        additional_instruction=(
            "The supporting page(s) should make clear the 30-minute PSAP notification requirement and indicate it is effective on or from April 15, 2025."
        ),
    )

    # 5) PSAP follow-up every 2 hours during ongoing outages
    psap_2h = evaluator.add_parallel(
        id="PSAPFollowUpEvery2Hours",
        desc="PSAP follow-up: every 2 hours during ongoing 911-impacting outages.",
        parent=req,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_text(ext.psap_2h_followup_statement),
        id="PSAP2HourFollowUpProcedureClaim",
        desc="The answer explicitly states the provider’s procedure/compliance is to provide follow-up PSAP notifications every 2 hours during an ongoing 911-impacting outage.",
        parent=psap_2h,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_urls(ext.psap_2h_followup_urls),
        id="PSAP2HourFollowUpEvidenceURLsProvided",
        desc="At least one reliable URL is provided that directly supports the 2-hour PSAP follow-up procedure/requirement claim.",
        parent=psap_2h,
        critical=True,
    )
    psap_2h_ev = evaluator.add_leaf(
        id="PSAP2HourFollowUpEvidenceURL",
        desc="At least one reliable URL is provided that directly supports the 2-hour PSAP follow-up procedure/requirement claim.",
        parent=psap_2h,
        critical=True,
    )
    await evaluator.verify(
        claim="The provider provides follow-up PSAP notifications at least every 2 hours during an ongoing 911-impacting outage.",
        node=psap_2h_ev,
        sources=ext.psap_2h_followup_urls,
        additional_instruction="The supporting page(s) should explicitly mention a follow-up cadence of every two hours (or similar wording) during ongoing outages.",
    )

    # 6) Tower inspection frequency standards
    towers = evaluator.add_parallel(
        id="TowerInspectionFrequencyStandards",
        desc="Tower inspection frequency standards per the stated constraints, with evidence.",
        parent=req,
        critical=True,
    )

    # 6a) Guyed towers: at least every 3 years
    guyed = evaluator.add_parallel(
        id="GuyedTowerInspectionAtLeastEvery3Years",
        desc="Guyed towers inspected at least once every three years.",
        parent=towers,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_text(ext.guyed_3y_statement),
        id="GuyedInspectionFrequencyClaim",
        desc="The answer explicitly states guyed cell towers are inspected at least once every three years.",
        parent=guyed,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_urls(ext.guyed_3y_urls),
        id="GuyedInspectionEvidenceURLsProvided",
        desc="At least one reliable URL is provided that directly supports the guyed-tower 3-year inspection frequency claim.",
        parent=guyed,
        critical=True,
    )
    guyed_ev = evaluator.add_leaf(
        id="GuyedInspectionEvidenceURL",
        desc="At least one reliable URL is provided that directly supports the guyed-tower 3-year inspection frequency claim.",
        parent=guyed,
        critical=True,
    )
    await evaluator.verify(
        claim="Guyed towers are inspected at least once every three years.",
        node=guyed_ev,
        sources=ext.guyed_3y_urls,
        additional_instruction="The supporting page(s) should clearly indicate a minimum inspection interval of once every 3 years for guyed towers.",
    )

    # 6b) Self-supporting towers: at least every 5 years
    selfsup = evaluator.add_parallel(
        id="SelfSupportingTowerInspectionAtLeastEvery5Years",
        desc="Self-supporting towers inspected at least once every five years.",
        parent=towers,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_text(ext.selfsupport_5y_statement),
        id="SelfSupportingInspectionFrequencyClaim",
        desc="The answer explicitly states self-supporting cell towers are inspected at least once every five years.",
        parent=selfsup,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_urls(ext.selfsupport_5y_urls),
        id="SelfSupportingInspectionEvidenceURLsProvided",
        desc="At least one reliable URL is provided that directly supports the self-supporting-tower 5-year inspection frequency claim.",
        parent=selfsup,
        critical=True,
    )
    selfsup_ev = evaluator.add_leaf(
        id="SelfSupportingInspectionEvidenceURL",
        desc="At least one reliable URL is provided that directly supports the self-supporting-tower 5-year inspection frequency claim.",
        parent=selfsup,
        critical=True,
    )
    await evaluator.verify(
        claim="Self-supporting towers are inspected at least once every five years.",
        node=selfsup_ev,
        sources=ext.selfsupport_5y_urls,
        additional_instruction="The supporting page(s) should clearly indicate a minimum inspection interval of once every 5 years for self-supporting towers.",
    )


async def build_optional_section(
    evaluator: Evaluator,
    parent,
    ext: ProviderComplianceExtraction,
) -> None:
    """
    Optional items: include if available; when included, provide supporting URLs.
    """
    opt = evaluator.add_parallel(
        id="OptionalAdditionalInformation",
        desc="Optional items requested: include if available; when included, provide supporting URLs.",
        parent=parent,
        critical=False,
    )

    # Optional: Backup power technology used
    tech = evaluator.add_sequential(
        id="BackupPowerTechnologyUsed",
        desc="Optional: backup power technology used (generators, batteries, fuel cells, or combination) with evidence if stated.",
        parent=opt,
        critical=False,
    )
    evaluator.add_custom_node(
        result=has_text(ext.backup_power_tech_statement),
        id="BackupPowerTechnologyClaim",
        desc="If provided, the answer states the backup power technology used.",
        parent=tech,
        critical=False,
    )
    evaluator.add_custom_node(
        result=has_urls(ext.backup_power_tech_urls) if has_text(ext.backup_power_tech_statement) else True,
        id="BackupPowerTechnologyEvidenceURLsProvided",
        desc="If the technology is stated, at least one reliable URL is provided that supports it.",
        parent=tech,
        critical=False,
    )
    tech_ev = evaluator.add_leaf(
        id="BackupPowerTechnologyEvidenceURL",
        desc="If the technology is stated, at least one reliable URL is provided that supports it.",
        parent=tech,
        critical=False,
    )
    if has_text(ext.backup_power_tech_statement):
        await evaluator.verify(
            claim=f"The provider uses the following backup power technology (or combination): {ext.backup_power_tech_statement}.",
            node=tech_ev,
            sources=ext.backup_power_tech_urls,
            additional_instruction="The supporting page(s) should clearly mention the stated technology (e.g., generators, batteries, fuel cells).",
        )
    else:
        # No claim provided -> skip evidence check
        tech_ev.score = 0.0
        tech_ev.status = "skipped"

    # Optional: Preventive maintenance / inspection schedule details
    pm = evaluator.add_sequential(
        id="PreventiveMaintenanceProgram",
        desc="Optional: tower inspection schedule and/or preventive maintenance program details, with evidence if stated.",
        parent=opt,
        critical=False,
    )
    evaluator.add_custom_node(
        result=has_text(ext.preventive_maintenance_statement),
        id="MaintenanceProgramClaim",
        desc="If provided, the answer describes the preventive maintenance program and/or inspection schedule at a high level.",
        parent=pm,
        critical=False,
    )
    evaluator.add_custom_node(
        result=has_urls(ext.preventive_maintenance_urls) if has_text(ext.preventive_maintenance_statement) else True,
        id="MaintenanceProgramEvidenceURLsProvided",
        desc="If such details are stated, at least one reliable URL is provided that supports them.",
        parent=pm,
        critical=False,
    )
    pm_ev = evaluator.add_leaf(
        id="MaintenanceProgramEvidenceURL",
        desc="If such details are stated, at least one reliable URL is provided that supports them.",
        parent=pm,
        critical=False,
    )
    if has_text(ext.preventive_maintenance_statement):
        await evaluator.verify(
            claim=f"The provider's preventive maintenance / inspection program includes: {ext.preventive_maintenance_statement}.",
            node=pm_ev,
            sources=ext.preventive_maintenance_urls,
            additional_instruction="The supporting page(s) should substantiate the described maintenance/inspection details.",
        )
    else:
        pm_ev.score = 0.0
        pm_ev.status = "skipped"


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
    Evaluate the answer for telecom provider Florida operations and FCC/NORS/PSAP compliance.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Provider identification must succeed before other parts are meaningful
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

    # Extract structured information from the answer
    ext: ProviderComplianceExtraction = await evaluator.extract(
        prompt=prompt_extract_provider_compliance(),
        template_class=ProviderComplianceExtraction,
        extraction_name="provider_compliance_extraction",
    )

    # Build verification tree according to rubric
    await build_provider_identification_section(evaluator, root, ext)
    await build_requirements_section(evaluator, root, ext)
    await build_optional_section(evaluator, root, ext)

    # Return structured summary
    return evaluator.get_summary()