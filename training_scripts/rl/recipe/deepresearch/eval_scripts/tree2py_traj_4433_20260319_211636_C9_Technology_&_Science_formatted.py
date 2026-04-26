import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_wireless_emergency_compliance"
TASK_DESCRIPTION = (
    "I am researching wireless carriers' emergency preparedness capabilities for a disaster resilience planning project. "
    "I need to identify four major facilities-based mobile wireless carriers in the United States that demonstrate comprehensive "
    "compliance with federal telecommunications emergency requirements.\n\n"
    "For each of the four carriers, please verify and provide documentation for the following compliance areas:\n\n"
    "1. Basic Qualification: Confirm the carrier is a facilities-based mobile wireless provider that offers consumer services across multiple US states or nationwide.\n\n"
    "2. FCC Backup Power Compliance: Verify that the carrier offers backup power solutions meeting FCC requirements under 47 CFR § 9.20, including:\n"
    "   - At least one option providing a minimum of 8 hours of standby backup power\n"
    "   - At least one option providing a minimum of 24 hours of standby backup power (as required since February 13, 2019)\n\n"
    "3. Mandatory Disaster Response Initiative (MDRI): Confirm that the carrier:\n"
    "   - Participates in the FCC's Mandatory Disaster Response Initiative\n"
    "   - Has established bilateral roaming agreements with other facilities-based mobile wireless providers for emergency situations\n\n"
    "4. Emergency Alert Systems: Verify that the carrier:\n"
    "   - Participates in the Wireless Emergency Alerts (WEA) system\n"
    "   - Is subject to FCC Network Outage Reporting System (NORS) requirements\n\n"
    "For each carrier and each compliance area, provide the carrier's name, a description of how they meet the requirement, "
    "and a direct reference URL from the carrier's official website or official FCC documentation that supports the compliance claim."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BasicQualification(BaseModel):
    facilities_based_desc: Optional[str] = None
    coverage_desc: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BackupPowerCompliance(BaseModel):
    option_8h_desc: Optional[str] = None
    option_24h_desc: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MDRICompliance(BaseModel):
    participation_desc: Optional[str] = None
    bilateral_roaming_desc: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EmergencySystemsCompliance(BaseModel):
    wea_desc: Optional[str] = None
    nors_desc: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CarrierCompliance(BaseModel):
    name: Optional[str] = None
    basic: Optional[BasicQualification] = None
    backup_power: Optional[BackupPowerCompliance] = None
    mdri: Optional[MDRICompliance] = None
    emergency: Optional[EmergencySystemsCompliance] = None


class ComplianceExtraction(BaseModel):
    carriers: List[CarrierCompliance] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_carriers() -> str:
    return """
    Extract from the answer up to four (4) facilities-based mobile wireless carriers in the United States and, for each, the compliance evidence requested.
    For each carrier mentioned, extract:
    - name: The carrier’s name (e.g., Verizon, AT&T, T-Mobile, UScellular)
    - basic: Information and sources supporting that the carrier is a facilities-based mobile wireless provider and offers consumer services across multiple U.S. states or nationwide.
      * facilities_based_desc: A short description quoted or closely paraphrased from the answer about facilities-based status
      * coverage_desc: A short description about consumer services coverage across multiple states or nationwide
      * sources: A list of URLs (official carrier website or official FCC documentation) cited in the answer that support facilities-based status and/or coverage
    - backup_power: Information and sources supporting compliance with FCC backup power requirements under 47 CFR § 9.20
      * option_8h_desc: A short description that an option providing at least 8 hours of standby power is offered
      * option_24h_desc: A short description that an option providing at least 24 hours of standby power is offered
      * sources: A list of URLs cited in the answer that support the backup power offerings and/or FCC 9.20 compliance
    - mdri: Information and sources supporting participation in the Mandatory Disaster Response Initiative (MDRI) and bilateral emergency roaming agreements
      * participation_desc: A short description that the carrier participates in the FCC’s MDRI (or its equivalent terminology)
      * bilateral_roaming_desc: A short description that the carrier has bilateral roaming agreements for emergency situations
      * sources: A list of URLs cited in the answer that support MDRI participation and/or roaming agreements
    - emergency: Information and sources supporting participation in Wireless Emergency Alerts (WEA) and that the carrier is subject to NORS
      * wea_desc: A short description that the carrier participates in WEA
      * nors_desc: A short description that the carrier is subject to FCC NORS outage reporting requirements
      * sources: A list of URLs cited in the answer that support WEA participation and/or NORS status

    Important:
    - Only extract URLs explicitly present in the answer (plain URLs or in markdown). Do not fabricate or infer URLs.
    - Prefer URLs from the carrier’s official site or official FCC documentation.
    - If any field lacks information in the answer, set it to null (or an empty list for sources).

    Return a JSON object:
    {
      "carriers": [
        {
          "name": "...",
          "basic": {
            "facilities_based_desc": "...",
            "coverage_desc": "...",
            "sources": ["...", "..."]
          },
          "backup_power": {
            "option_8h_desc": "...",
            "option_24h_desc": "...",
            "sources": ["...", "..."]
          },
          "mdri": {
            "participation_desc": "...",
            "bilateral_roaming_desc": "...",
            "sources": ["...", "..."]
          },
          "emergency": {
            "wea_desc": "...",
            "nors_desc": "...",
            "sources": ["...", "..."]
          }
        }
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls and any(isinstance(u, str) and u.strip() for u in urls))


async def _verify_leaf_with_sources(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent,
    critical: bool,
    claim: str,
    sources: Optional[List[str]],
    add_ins: str = "None",
) -> Tuple[bool, str]:
    """
    Create a leaf node and verify the claim using provided sources.
    If sources are missing/empty, mark the leaf as failed without calling the verifier.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    if not _has_sources(sources):
        leaf.score = 0.0
        leaf.status = "failed"
        return False, "no_sources"
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=add_ins,
    )
    return (leaf.score == 1.0), "verified"


# --------------------------------------------------------------------------- #
# Carrier subtree builders                                                    #
# --------------------------------------------------------------------------- #
async def build_basic_qualification_subtree(
    evaluator: Evaluator,
    *,
    parent,
    idx: int,
    carrier: CarrierCompliance
):
    carrier_name = carrier.name or f"Carrier #{idx+1}"

    basic_node = evaluator.add_parallel(
        id=f"carrier_{idx+1}_basic_qualification",
        desc="Carrier meets basic qualification requirements as a facilities-based provider offering consumer services",
        parent=parent,
        critical=True,
    )

    provider_status_node = evaluator.add_parallel(
        id=f"carrier_{idx+1}_provider_status",
        desc="Verification of carrier's facilities-based provider status and service scope",
        parent=basic_node,
        critical=True,
    )

    basic_sources = (carrier.basic.sources if carrier.basic else []) if carrier.basic else []

    # Facilities-based provider verification
    await _verify_leaf_with_sources(
        evaluator,
        node_id=f"carrier_{idx+1}_facilities_based_provider",
        desc="Carrier is identified as a facilities-based mobile wireless provider operating network infrastructure in the United States",
        parent=provider_status_node,
        critical=True,
        claim="This page confirms that the company is a facilities-based mobile wireless provider that operates its own network infrastructure in the United States (as opposed to being solely an MVNO).",
        sources=basic_sources,
        add_ins="Accept explicit statements like 'facilities-based', 'operates its own wireless network', or equivalent. Pages from the carrier or the FCC are acceptable.",
    )

    # Consumer services coverage verification
    await _verify_leaf_with_sources(
        evaluator,
        node_id=f"carrier_{idx+1}_consumer_services_coverage",
        desc="Carrier offers commercial wireless services to consumers across multiple US states or nationwide",
        parent=provider_status_node,
        critical=True,
        claim="This page shows that the company offers commercial mobile wireless services to consumers across multiple U.S. states or nationwide.",
        sources=basic_sources,
        add_ins="Coverage maps, nationwide service statements, or multi-state availability pages count. Carrier or FCC pages are acceptable.",
    )

    # Reference URL verification for this section
    await _verify_leaf_with_sources(
        evaluator,
        node_id=f"carrier_{idx+1}_qualification_reference_url",
        desc="Valid reference URL documenting carrier's facilities-based status and service coverage",
        parent=basic_node,
        critical=True,
        claim="This is an official page (from the carrier or the U.S. FCC) that provides documentation supporting the carrier's facilities-based status and/or multi-state/nationwide consumer service coverage.",
        sources=basic_sources,
        add_ins="The URL should be from the carrier's domain or an FCC domain. It should include relevant language about facilities-based operation and/or widespread coverage.",
    )


async def build_backup_power_subtree(
    evaluator: Evaluator,
    *,
    parent,
    idx: int,
    carrier: CarrierCompliance
):
    backup_node = evaluator.add_parallel(
        id=f"carrier_{idx+1}_fcc_backup_power_compliance",
        desc="Carrier complies with FCC backup power requirements under 47 CFR § 9.20",
        parent=parent,
        critical=True,
    )

    options_node = evaluator.add_parallel(
        id=f"carrier_{idx+1}_backup_power_options",
        desc="Verification of backup power duration options offered to customers",
        parent=backup_node,
        critical=True,
    )

    bp_sources = (carrier.backup_power.sources if carrier.backup_power else []) if carrier.backup_power else []

    # 8-hour option
    await _verify_leaf_with_sources(
        evaluator,
        node_id=f"carrier_{idx+1}_eight_hour_option",
        desc="Carrier offers customers at least one backup power option providing minimum 8 hours of standby power for premises equipment",
        parent=options_node,
        critical=True,
        claim="This page shows that at least one available backup power option provides a minimum of 8 hours of standby backup power for customer premises equipment used for voice service (per 47 CFR § 9.20).",
        sources=bp_sources,
        add_ins="Accept language such as '8 hours of standby', '8-hour battery', or equivalent. Program pages for home phone, fiber voice, or CPE battery backups are acceptable if they specify duration.",
    )

    # 24-hour option
    await _verify_leaf_with_sources(
        evaluator,
        node_id=f"carrier_{idx+1}_twenty_four_hour_option",
        desc="Carrier offers customers at least one backup power option providing minimum 24 hours of standby power for premises equipment (required since February 13, 2019)",
        parent=options_node,
        critical=True,
        claim="This page shows that at least one available backup power option provides a minimum of 24 hours of standby backup power for customer premises equipment used for voice service (as required since February 13, 2019) (per 47 CFR § 9.20).",
        sources=bp_sources,
        add_ins="Accept language such as '24 hours', '24-hour battery', 'two 12-hour batteries', or equivalent statements indicating at least 24 hours total standby power.",
    )

    # Reference URL
    await _verify_leaf_with_sources(
        evaluator,
        node_id=f"carrier_{idx+1}_backup_power_reference_url",
        desc="Valid reference URL documenting carrier's backup power offerings and FCC compliance",
        parent=backup_node,
        critical=True,
        claim="This is an official carrier page or U.S. FCC page documenting the carrier’s backup power offerings and/or compliance with FCC 47 CFR § 9.20 backup power requirements.",
        sources=bp_sources,
        add_ins="Program details pages, customer notices, or FCC compliance pages are acceptable if they mention backup power durations or 47 CFR § 9.20.",
    )


async def build_mdri_subtree(
    evaluator: Evaluator,
    *,
    parent,
    idx: int,
    carrier: CarrierCompliance
):
    mdri_node = evaluator.add_parallel(
        id=f"carrier_{idx+1}_mdri_disaster_response",
        desc="Carrier participates in the Mandatory Disaster Response Initiative (MDRI) as required by FCC regulations",
        parent=parent,
        critical=True,
    )

    req_node = evaluator.add_parallel(
        id=f"carrier_{idx+1}_mdri_requirements",
        desc="Verification of carrier's MDRI participation and roaming agreements",
        parent=mdri_node,
        critical=True,
    )

    mdri_sources = (carrier.mdri.sources if carrier.mdri else []) if carrier.mdri else []

    # MDRI participation
    await _verify_leaf_with_sources(
        evaluator,
        node_id=f"carrier_{idx+1}_mdri_participation",
        desc="Carrier is confirmed to participate in the FCC's Mandatory Disaster Response Initiative for network resilience during disasters",
        parent=req_node,
        critical=True,
        claim="This page confirms participation in the FCC's Mandatory Disaster Response Initiative (MDRI) or its equivalent disaster resiliency framework for wireless carriers.",
        sources=mdri_sources,
        add_ins="Accept explicit mentions of 'Mandatory Disaster Response Initiative (MDRI)'. Also accept references to the 'Wireless Resiliency Cooperative Framework' if presented as the FCC-mandated program.",
    )

    # Bilateral roaming agreements
    await _verify_leaf_with_sources(
        evaluator,
        node_id=f"carrier_{idx+1}_bilateral_roaming_agreements",
        desc="Carrier has established bilateral roaming agreements with other facilities-based mobile wireless providers for emergency situations and disaster response",
        parent=req_node,
        critical=True,
        claim="This page indicates the existence of bilateral roaming agreements or roaming arrangements with other facilities-based mobile wireless providers for emergencies/disasters.",
        sources=mdri_sources,
        add_ins="Look for language like 'roaming during disasters', 'roaming partner agreements', 'mutual aid roaming', or similar bilateral arrangements.",
    )

    # Reference URL
    await _verify_leaf_with_sources(
        evaluator,
        node_id=f"carrier_{idx+1}_mdri_reference_url",
        desc="Valid reference URL documenting carrier's MDRI participation and disaster response policies",
        parent=mdri_node,
        critical=True,
        claim="This is an official carrier or U.S. FCC page that documents MDRI participation and/or disaster response/roaming policies.",
        sources=mdri_sources,
        add_ins="FCC press releases or orders, or carrier policy/disaster response pages are acceptable evidence.",
    )


async def build_emergency_systems_subtree(
    evaluator: Evaluator,
    *,
    parent,
    idx: int,
    carrier: CarrierCompliance
):
    emer_node = evaluator.add_parallel(
        id=f"carrier_{idx+1}_emergency_alert_systems",
        desc="Carrier participates in federally mandated emergency alert and reporting systems",
        parent=parent,
        critical=True,
    )

    part_node = evaluator.add_parallel(
        id=f"carrier_{idx+1}_alert_system_participation",
        desc="Verification of carrier's participation in WEA system and NORS compliance",
        parent=emer_node,
        critical=True,
    )

    emer_sources = (carrier.emergency.sources if carrier.emergency else []) if carrier.emergency else []

    # WEA participation
    await _verify_leaf_with_sources(
        evaluator,
        node_id=f"carrier_{idx+1}_wea_participation",
        desc="Carrier participates in the Wireless Emergency Alerts (WEA) system and can transmit emergency messages from authorized government authorities",
        parent=part_node,
        critical=True,
        claim="This page confirms that the carrier participates in the Wireless Emergency Alerts (WEA) system and can transmit alerts issued by authorized public safety authorities.",
        sources=emer_sources,
        add_ins="Accept explicit statements that the carrier supports or participates in WEA (CMAS).",
    )

    # NORS subject status
    await _verify_leaf_with_sources(
        evaluator,
        node_id=f"carrier_{idx+1}_nors_subject_status",
        desc="Carrier is subject to FCC Network Outage Reporting System (NORS) requirements for reporting service disruptions",
        parent=part_node,
        critical=True,
        claim="This page confirms that the carrier (as a facilities-based mobile wireless provider) is subject to the FCC's Network Outage Reporting System (NORS) reporting requirements.",
        sources=emer_sources,
        add_ins="Accept FCC rules/orders or carrier compliance pages indicating NORS applicability to mobile/wireless carriers. Generic FCC pages are acceptable if they clearly impose NORS on wireless carriers.",
    )

    # Reference URL
    await _verify_leaf_with_sources(
        evaluator,
        node_id=f"carrier_{idx+1}_emergency_systems_reference_url",
        desc="Valid reference URL documenting carrier's WEA participation and NORS compliance status",
        parent=emer_node,
        critical=True,
        claim="This is an official carrier or U.S. FCC page that documents WEA participation and/or NORS obligations for the carrier.",
        sources=emer_sources,
        add_ins="Either a carrier page stating WEA participation or an FCC page indicating WEA or NORS applicability to the carrier satisfies this.",
    )


async def verify_carrier(
    evaluator: Evaluator,
    *,
    parent,
    idx: int,
    carrier: CarrierCompliance
):
    carrier_name = carrier.name or f"Carrier #{idx+1}"
    carrier_node = evaluator.add_parallel(
        id=f"carrier_{idx+1}",
        desc=f"Wireless carrier #{idx+1} meeting emergency preparedness criteria",
        parent=parent,
        critical=False,  # Allow partial credit per-carrier
    )

    # Build subtrees per rubric
    await build_basic_qualification_subtree(evaluator, parent=carrier_node, idx=idx, carrier=carrier)
    await build_backup_power_subtree(evaluator, parent=carrier_node, idx=idx, carrier=carrier)
    await build_mdri_subtree(evaluator, parent=carrier_node, idx=idx, carrier=carrier)
    await build_emergency_systems_subtree(evaluator, parent=carrier_node, idx=idx, carrier=carrier)


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
    Evaluate an answer for the US wireless carriers emergency preparedness compliance task.
    """
    evaluator = Evaluator()
    # Important: root set to non-critical to avoid framework constraint that all children of a critical node must be critical.
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

    # Extract structured compliance info
    extraction = await evaluator.extract(
        prompt=prompt_extract_carriers(),
        template_class=ComplianceExtraction,
        extraction_name="carriers_compliance_extraction",
    )

    # Prepare up to 4 carriers, pad if needed
    carriers: List[CarrierCompliance] = list(extraction.carriers)[:4]
    while len(carriers) < 4:
        carriers.append(CarrierCompliance())

    # Optional: record a quick summary of extracted carrier names
    evaluator.add_custom_info(
        info={"extracted_carriers": [c.name for c in carriers]},
        info_type="extraction_summary",
        info_name="extracted_carrier_names",
    )

    # Build verification subtrees for each carrier
    tasks = []
    for i in range(4):
        tasks.append(verify_carrier(evaluator, parent=root, idx=i, carrier=carriers[i]))
    # Run sequentially to keep logs tidy; could be gathered if desired
    for t in tasks:
        await t

    return evaluator.get_summary()