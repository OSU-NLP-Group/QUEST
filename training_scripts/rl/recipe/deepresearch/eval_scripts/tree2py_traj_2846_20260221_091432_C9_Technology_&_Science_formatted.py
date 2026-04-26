import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.verification_tree import VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_major_carriers_infrastructure_assessment"
TASK_DESCRIPTION = """
Identify the three major US wireless telecommunications carriers that meet ALL of the following comprehensive technical, regulatory, and infrastructure criteria as of December 31, 2025:

1. 5G Standalone Deployment: The carrier must have deployed a 5G Standalone (SA) network with nationwide coverage in the United States by October 31, 2025, with the specific deployment month and year publicly documented.

2. FCC Regulatory Compliance: The carrier must be subject to FCC Network Outage Reporting System (NORS) requirements, including the threshold for reporting outages affecting 30,000 or more users, submitting initial reports within 72 hours, final reports within 30 days, and notifying PSAPs within 30 minutes of 911-impacting outages.

3. Backup Power Infrastructure: The carrier must maintain emergency backup power meeting FCC requirements: minimum 24 hours for central office assets and minimum 8 hours for cell sites.

4. Multi-Band Spectrum Deployment: The carrier must have deployed both low-band spectrum (below 1 GHz, such as 600 MHz or 700 MHz) and mid-band spectrum (1-6 GHz range, such as C-band) for its 5G network.

5. Network Reliability Standards: The carrier should target five nines (99.999%) network reliability, representing maximum 5.26 minutes of downtime per year, consistent with telecommunications industry gold standards.

6. Subscriber Base and Market Position: The carrier must serve over 100 million subscribers as of Q4 2025 and rank among the top three nationwide wireless carriers in the United States by subscriber count.

For each identified carrier, provide: (1) the carrier's name, (2) the specific month and year of their nationwide 5G SA deployment, (3) their subscriber count as of Q4 2025, (4) specific low-band and mid-band spectrum bands they have deployed for 5G, and (5) supporting URL references for each major category of information.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CarrierExtraction(BaseModel):
    name: Optional[str] = None

    # 5G SA deployment
    sa_deployment_month_year: Optional[str] = None  # e.g., "August 2023"
    sa_urls: List[str] = Field(default_factory=list)

    # FCC NORS / outage notification
    nors_urls: List[str] = Field(default_factory=list)

    # Backup power
    backup_power_urls: List[str] = Field(default_factory=list)

    # Spectrum deployment
    low_band_5g_bands: List[str] = Field(default_factory=list)  # e.g., ["600 MHz", "700 MHz"]
    mid_band_5g_bands: List[str] = Field(default_factory=list)  # e.g., ["C-band", "3.7 GHz"]
    spectrum_urls: List[str] = Field(default_factory=list)

    # Subscriber base / ranking
    subscriber_count_q4_2025: Optional[str] = None  # Keep as string to be flexible
    subscriber_urls: List[str] = Field(default_factory=list)

    # Network layers and NOC
    network_layers_urls: List[str] = Field(default_factory=list)
    noc_urls: List[str] = Field(default_factory=list)

    # Reliability target (optional "should")
    five_nines_claim: Optional[bool] = None  # True if answer explicitly claims a 99.999% target
    reliability_urls: List[str] = Field(default_factory=list)


class CarriersExtraction(BaseModel):
    carriers: List[CarrierExtraction] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_carriers() -> str:
    return """
    Extract up to three US nationwide wireless carriers discussed in the answer that the answer claims meet the criteria.
    For each carrier, return an object with the following fields:

    - name: The carrier name (e.g., "Verizon", "AT&T", "T-Mobile").
    - sa_deployment_month_year: The specific month and year for the carrier's nationwide 5G Standalone (SA) deployment, as written (e.g., "August 2023"). If not stated, null.
    - sa_urls: A list of URL(s) used in the answer to support the nationwide 5G SA deployment timing (month/year) and/or nationwide SA coverage.
    - nors_urls: A list of URL(s) used in the answer to support the FCC NORS / PSAP notification requirements for this carrier (can be FCC pages that apply to wireless providers generally).
    - backup_power_urls: A list of URL(s) used in the answer to support FCC backup power requirements and/or the carrier’s compliance/statement.
    - low_band_5g_bands: A list of specific low-band (<1 GHz) spectrum bands that the answer claims the carrier uses for 5G (e.g., "600 MHz", "700 MHz").
    - mid_band_5g_bands: A list of specific mid-band (1–6 GHz) spectrum bands that the answer claims the carrier uses for 5G (e.g., "C-band", "3.7 GHz").
    - spectrum_urls: A list of URL(s) used in the answer to support the carrier’s low-band and mid-band 5G deployment/usage.
    - subscriber_count_q4_2025: The subscriber count as of Q4 2025 as presented in the answer, in string form (e.g., "115 million"). If not given, null.
    - subscriber_urls: A list of URL(s) used in the answer to support the Q4 2025 subscriber count and/or ranking.
    - network_layers_urls: A list of URL(s) used in the answer to support that the carrier’s network includes RAN, transport network, and core network.
    - noc_urls: A list of URL(s) used in the answer to support that the carrier operates a 24/7 Network Operations Center (NOC).
    - five_nines_claim: true if the answer explicitly claims a "five nines" (99.999%) reliability target for the carrier; false otherwise; null if unclear.
    - reliability_urls: A list of URL(s) used in the answer to support the five nines reliability target (if claimed).

    NOTES:
    - Only extract URLs that appear in the answer text. If no URL is provided for a category, return an empty list.
    - Keep numbers and dates as strings exactly as written in the answer.
    - Return a JSON object with a top-level field "carriers" as an array with up to three carrier objects.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_carrier(
    evaluator: Evaluator,
    parent: VerificationNode,
    carrier: Optional[CarrierExtraction],
    index: int
) -> None:
    """
    Build verification subtree for one carrier and run verifications.
    """
    # Parent node for this carrier (non-critical to allow partial credit across carriers)
    carrier_node = evaluator.add_parallel(
        id=f"carrier_{index+1}",
        desc=f"{index+1}st/nd/rd identified carrier (index {index})",
        parent=parent,
        critical=False
    )

    # Carrier name provided (critical leaf as custom existence check)
    name_exists = carrier is not None and carrier.name is not None and carrier.name.strip() != ""
    evaluator.add_custom_node(
        result=name_exists,
        id=f"c{index}_carrier_name_provided",
        desc="Carrier name is provided.",
        parent=carrier_node,
        critical=True
    )

    # ------------------------- 5G SA Deployment ------------------------- #
    sa_node = evaluator.add_parallel(
        id=f"c{index}_5g_sa_deployment",
        desc="Nationwide 5G Standalone (SA) deployment timing and evidence.",
        parent=carrier_node,
        critical=True
    )

    sa_urls = (carrier.sa_urls if carrier else []) if carrier else []
    sa_month_year = (carrier.sa_deployment_month_year if carrier else None)

    # URL provided (critical existence)
    evaluator.add_custom_node(
        result=bool(sa_urls),
        id=f"c{index}_sa_deploy_url_provided",
        desc="Provides a supporting URL documenting the nationwide 5G SA deployment month/year.",
        parent=sa_node,
        critical=True
    )

    # Month/Year provided (critical existence)
    evaluator.add_custom_node(
        result=bool(sa_month_year and sa_month_year.strip()),
        id=f"c{index}_sa_deploy_month_year_provided",
        desc="Provides the specific month and year for nationwide 5G SA deployment.",
        parent=sa_node,
        critical=True
    )

    # Nationwide 5G SA by Oct 31, 2025 (critical factual)
    sa_by_date_leaf = evaluator.add_leaf(
        id=f"c{index}_sa_by_2025_10_31",
        desc="Carrier has nationwide US 5G Standalone (SA) deployed by Oct 31, 2025.",
        parent=sa_node,
        critical=True
    )
    sa_claim_name = carrier.name if name_exists else "the carrier"
    sa_claim = (
        f"By October 31, 2025, {sa_claim_name} had deployed a nationwide 5G Standalone (SA) network in the United States."
    )
    await evaluator.verify(
        claim=sa_claim,
        node=sa_by_date_leaf,
        sources=sa_urls,
        additional_instruction="Verify that the source(s) explicitly indicate a nationwide 5G Standalone (SA) network, not Non-Standalone (NSA), and that this status was achieved on or before Oct 31, 2025. Allow equivalent phrasing like 'national', 'nationwide', or 'available nationwide'."
    )

    # ------------------------- FCC NORS Compliance ---------------------- #
    nors_node = evaluator.add_parallel(
        id=f"c{index}_fcc_nors_compliance",
        desc="FCC NORS/notification requirements applicability and evidence.",
        parent=carrier_node,
        critical=True
    )

    nors_urls = (carrier.nors_urls if carrier else [])

    # FCC NORS URL provided (critical existence)
    evaluator.add_custom_node(
        result=bool(nors_urls),
        id=f"c{index}_nors_url_provided",
        desc="Provides a supporting URL for the FCC NORS/notification requirements used in the answer.",
        parent=nors_node,
        critical=True
    )

    # Subject to NORS (critical factual)
    subject_to_nors_leaf = evaluator.add_leaf(
        id=f"c{index}_subject_to_nors",
        desc="Carrier is subject to FCC Network Outage Reporting System (NORS) requirements.",
        parent=nors_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{sa_claim_name} is subject to FCC Network Outage Reporting System (NORS) outage reporting requirements.",
        node=subject_to_nors_leaf,
        sources=nors_urls,
        additional_instruction="Support can come from FCC pages stating that wireless communications providers are subject to NORS requirements."
    )

    # NORS 30,000 user threshold (critical factual)
    nors_30k_leaf = evaluator.add_leaf(
        id=f"c{index}_nors_30000_threshold",
        desc="Includes the NORS reporting threshold for outages affecting 30,000 or more users.",
        parent=nors_node,
        critical=True
    )
    await evaluator.verify(
        claim="FCC NORS requires reporting outages that affect 30,000 or more users.",
        node=nors_30k_leaf,
        sources=nors_urls,
        additional_instruction="Confirm that the FCC threshold is 30,000 users for reportable outages (or equivalently worded thresholds in official FCC rules/FAQs/public notices)."
    )

    # NORS initial report within 72 hours (critical factual)
    nors_72h_leaf = evaluator.add_leaf(
        id=f"c{index}_nors_initial_72h",
        desc="Includes the requirement to submit initial outage reports within 72 hours (3 calendar days) of discovering a reportable outage.",
        parent=nors_node,
        critical=True
    )
    await evaluator.verify(
        claim="FCC NORS requires an initial outage report within 72 hours (3 calendar days) of discovering a reportable outage.",
        node=nors_72h_leaf,
        sources=nors_urls,
        additional_instruction="Look for explicit timing requirements for the initial NORS report."
    )

    # NORS final report within 30 days (critical factual)
    nors_30d_leaf = evaluator.add_leaf(
        id=f"c{index}_nors_final_30d",
        desc="Includes the requirement to submit final outage reports within 30 days after discovering the outage.",
        parent=nors_node,
        critical=True
    )
    await evaluator.verify(
        claim="FCC NORS requires a final outage report within 30 days after discovering the outage.",
        node=nors_30d_leaf,
        sources=nors_urls,
        additional_instruction="Look for explicit timing requirements for the final NORS report."
    )

    # PSAP notification within 30 minutes for 911-impacting outages (critical factual)
    psap_30m_leaf = evaluator.add_leaf(
        id=f"c{index}_psap_notify_30m",
        desc="Includes the requirement to notify affected PSAPs within 30 minutes of discovering a 911-impacting outage.",
        parent=nors_node,
        critical=True
    )
    await evaluator.verify(
        claim="Providers must notify affected PSAPs within 30 minutes of discovering a 911-impacting outage.",
        node=psap_30m_leaf,
        sources=nors_urls,
        additional_instruction="Confirm that FCC rules require PSAP notification within 30 minutes for 911-impacting outages."
    )

    # ------------------------- Backup Power Infrastructure -------------- #
    backup_node = evaluator.add_parallel(
        id=f"c{index}_backup_power",
        desc="Backup power requirements and evidence.",
        parent=carrier_node,
        critical=True
    )

    backup_urls = (carrier.backup_power_urls if carrier else [])

    evaluator.add_custom_node(
        result=bool(backup_urls),
        id=f"c{index}_backup_power_url_provided",
        desc="Provides a supporting URL documenting the cited backup power requirement and/or carrier compliance basis used in the answer.",
        parent=backup_node,
        critical=True
    )

    backup_24h_leaf = evaluator.add_leaf(
        id=f"c{index}_backup_24h_central_office",
        desc="Meets (or states meeting) the minimum 24 hours backup power requirement for assets inside central offices.",
        parent=backup_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cited sources indicate that {sa_claim_name} meets (or states meeting) at least 24 hours of backup power for central office assets, consistent with FCC requirements.",
        node=backup_24h_leaf,
        sources=backup_urls,
        additional_instruction="Accept either explicit statements of carrier compliance or authoritative FCC requirement references paired with carrier applicability."
    )

    backup_8h_leaf = evaluator.add_leaf(
        id=f"c{index}_backup_8h_cell_sites",
        desc="Meets (or states meeting) the minimum 8 hours backup power requirement for cell sites.",
        parent=backup_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cited sources indicate that {sa_claim_name} meets (or states meeting) at least 8 hours of backup power for cell sites, consistent with FCC requirements.",
        node=backup_8h_leaf,
        sources=backup_urls,
        additional_instruction="Accept either explicit statements of carrier compliance or authoritative FCC requirement references paired with carrier applicability."
    )

    # ------------------------- Spectrum Deployment ---------------------- #
    spectrum_node = evaluator.add_parallel(
        id=f"c{index}_spectrum_deployment",
        desc="Low-band and mid-band 5G spectrum deployment and evidence.",
        parent=carrier_node,
        critical=True
    )

    spectrum_urls = (carrier.spectrum_urls if carrier else [])
    evaluator.add_custom_node(
        result=bool(spectrum_urls),
        id=f"c{index}_spectrum_url_provided",
        desc="Provides a supporting URL for the carrier’s low-band and mid-band 5G spectrum deployment/usage.",
        parent=spectrum_node,
        critical=True
    )

    low_bands = (carrier.low_band_5g_bands if carrier else [])
    mid_bands = (carrier.mid_band_5g_bands if carrier else [])

    low_band_leaf = evaluator.add_leaf(
        id=f"c{index}_low_band_5g_deployed",
        desc="Carrier has deployed low-band (<1 GHz) spectrum for 5G and names at least one specific low-band (e.g., 600 MHz or 700 MHz).",
        parent=spectrum_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{sa_claim_name} has deployed low-band (<1 GHz) spectrum for 5G (e.g., {low_bands}).",
        node=low_band_leaf,
        sources=spectrum_urls,
        additional_instruction="Confirm that at least one specific low-band like 600 MHz or 700 MHz is used for 5G by this carrier."
    )

    mid_band_leaf = evaluator.add_leaf(
        id=f"c{index}_mid_band_5g_deployed",
        desc="Carrier has deployed mid-band (1–6 GHz) spectrum for 5G and names at least one specific mid-band (e.g., C-band).",
        parent=spectrum_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{sa_claim_name} has deployed mid-band (1–6 GHz) spectrum for 5G (e.g., {mid_bands}).",
        node=mid_band_leaf,
        sources=spectrum_urls,
        additional_instruction="Confirm that at least one specific mid-band (e.g., C-band, 3.7 GHz) is used for 5G by this carrier."
    )

    # ------------------------- Subscriber Base & Market Position -------- #
    subs_node = evaluator.add_parallel(
        id=f"c{index}_subscriber_base_position",
        desc="Scale/top-3 status and evidence.",
        parent=carrier_node,
        critical=True
    )

    sub_urls = (carrier.subscriber_urls if carrier else [])
    evaluator.add_custom_node(
        result=bool(sub_urls),
        id=f"c{index}_subscriber_url_provided",
        desc="Provides a supporting URL for the Q4 2025 subscriber count and/or ranking claim used in the answer.",
        parent=subs_node,
        critical=True
    )

    # Subscriber count provided (existence)
    subs_count_exists = carrier is not None and carrier.subscriber_count_q4_2025 is not None and carrier.subscriber_count_q4_2025.strip() != ""
    evaluator.add_custom_node(
        result=subs_count_exists,
        id=f"c{index}_subscriber_count_provided",
        desc="Provides the subscriber count as of Q4 2025.",
        parent=subs_node,
        critical=True
    )

    # Over 100M (critical factual)
    over_100m_leaf = evaluator.add_leaf(
        id=f"c{index}_subscriber_over_100m",
        desc="Subscriber count as of Q4 2025 is over 100 million.",
        parent=subs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"As of Q4 2025, {sa_claim_name} served over 100 million subscribers.",
        node=over_100m_leaf,
        sources=sub_urls,
        additional_instruction="Consider 'over 100 million' satisfied if the cited figure is >= 100,000,000. Accept aggregate totals (e.g., postpaid + prepaid) if the source indicates total subscribers."
    )

    # Top three by subscribers (critical factual)
    top3_leaf = evaluator.add_leaf(
        id=f"c{index}_top_three_by_subs",
        desc="Ranks among the top three nationwide US wireless carriers by subscriber count.",
        parent=subs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"As of Q4 2025, {sa_claim_name} ranked among the top three nationwide US wireless carriers by subscriber count.",
        node=top3_leaf,
        sources=sub_urls,
        additional_instruction="Look for sources listing subscriber rankings or market share indicating this carrier is in the top three nationwide by subscribers."
    )

    # ------------------------- Network Layers Present ------------------- #
    layers_leaf = evaluator.add_leaf(
        id=f"c{index}_network_layers_present",
        desc="Carrier network includes the three essential network layers: Radio Access Network (RAN), Transport Network, and Core Network.",
        parent=carrier_node,
        critical=True
    )
    layers_urls = (carrier.network_layers_urls if carrier else [])
    await evaluator.verify(
        claim=f"{sa_claim_name}'s network includes Radio Access Network (RAN), transport/backhaul network, and core network.",
        node=layers_leaf,
        sources=layers_urls,
        additional_instruction="Accept standard telecom architecture descriptions from credible technical or carrier sources that clearly state RAN, transport (backhaul/middle mile), and core network components."
    )

    # ------------------------- NOC 24/7 --------------------------------- #
    noc_leaf = evaluator.add_leaf(
        id=f"c{index}_noc_24_7",
        desc="Carrier operates a Network Operations Center (NOC) for 24/7 network monitoring/management/maintenance.",
        parent=carrier_node,
        critical=True
    )
    noc_urls = (carrier.noc_urls if carrier else [])
    await evaluator.verify(
        claim=f"{sa_claim_name} operates a 24/7 Network Operations Center (NOC) for network monitoring, management, and maintenance.",
        node=noc_leaf,
        sources=noc_urls,
        additional_instruction="Accept sources that explicitly mention 24/7 NOC operations or continuous network monitoring/operations centers."
    )

    # ------------------------- Network Reliability Target (non-mandatory) #
    reliability_node = evaluator.add_parallel(
        id=f"c{index}_reliability_target",
        desc='Non-mandatory reliability target ("should").',
        parent=carrier_node,
        critical=False
    )

    five_leaf = evaluator.add_leaf(
        id=f"c{index}_five_nines_if_claimed",
        desc="If the answer claims a five-nines (99.999%) target (≤5.26 minutes downtime/year), it is stated consistently and supported by a URL.",
        parent=reliability_node,
        critical=False
    )

    if carrier and carrier.five_nines_claim:
        rel_urls = carrier.reliability_urls
        await evaluator.verify(
            claim=f"{sa_claim_name} targets 99.999% (five nines) network reliability (≈5.26 minutes of downtime per year).",
            node=five_leaf,
            sources=rel_urls,
            additional_instruction="The source should indicate a reliability target at or near 99.999% (five nines). Accept variations that clearly mean five nines and mention ~5.26 minutes/year."
        )
    else:
        # Not claimed -> mark as skipped (non-mandatory)
        five_leaf.score = 0.0
        five_leaf.status = "skipped"


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
    Evaluate an answer for the US major carriers infrastructure assessment task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Carriers evaluated independently for partial credit
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

    # IMPORTANT: Root is non-critical to allow non-critical children and partial credit
    # (If set to critical=True, the framework enforces all children critical, which conflicts with the rubric.)

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_carriers(),
        template_class=CarriersExtraction,
        extraction_name="carriers_extraction"
    )

    # Normalize to exactly three entries (pad with empty if fewer)
    carriers: List[CarrierExtraction] = list(extracted.carriers[:3])
    while len(carriers) < 3:
        carriers.append(CarrierExtraction())

    # Add top-level node to mirror rubric root description while remaining non-critical
    rubric_root = evaluator.add_parallel(
        id="US_Major_Carriers_Infrastructure_Assessment",
        desc="Evaluate up to three identified US wireless carriers against the stated technical, regulatory, infrastructure, and scale criteria; allow partial credit across carriers.",
        parent=root,
        critical=False
    )

    # Verify up to 3 carriers
    for i in range(3):
        await verify_carrier(
            evaluator=evaluator,
            parent=rubric_root,
            carrier=carriers[i],
            index=i
        )

    # Optional: log a summary of extracted carrier names
    evaluator.add_custom_info(
        info={"extracted_carrier_names": [c.name for c in carriers]},
        info_type="extraction_overview",
        info_name="extraction_overview"
    )

    return evaluator.get_summary()