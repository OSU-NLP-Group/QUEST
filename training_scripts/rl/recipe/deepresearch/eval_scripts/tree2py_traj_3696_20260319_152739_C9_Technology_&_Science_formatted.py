import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific metadata                                                      #
# --------------------------------------------------------------------------- #
TASK_ID = "technology_sector_analysis_2025"
TASK_DESCRIPTION = """Based on 2025 technology sector data, identify four specific technology entities that meet the following comprehensive requirements:

Entity 1 - Cloud Infrastructure Market Leader:
Identify the cloud infrastructure provider that held the highest market share in Q2 2025. The provider must offer Infrastructure as a Service (IaaS) to enterprise customers. Provide the provider's official name, their exact market share percentage in Q2 2025, and a reference URL confirming this market position.

Entity 2 - Major Technology Acquisition:
Identify a technology sector acquisition that was completed in 2025 or early 2026 with a deal value exceeding $10 billion. The acquiring company must be a major technology firm. Provide: (a) the names of both the acquiring company and the target company, (b) the exact deal value in billions of dollars, (c) the specific completion date, (d) reference URLs confirming the deal value and completion timeline.

Entity 3 - Advanced U.S. Semiconductor Fabrication Facility:
Identify a semiconductor fabrication facility located in the United States that meets all of the following criteria: (a) uses advanced process node technology of 10nm or below, (b) entered volume production by the end of 2024, (c) has a stated monthly wafer production capacity. Provide: the operating company name, specific U.S. state and city where located, the exact process node technology used (e.g., 4nm, 5nm), the wafer size, the monthly wafer capacity at full utilization, the date or timeframe when volume production began, and reference URLs confirming location, technical specifications, production capacity, and operational status.

Entity 4 - Commercial Quantum Computing System:
Identify a quantum computing system that was commercially launched or announced for commercial availability in 2025 or later, with at least 90 physical qubits. Provide: the system provider company name, the official system name or model designation, the exact number of physical qubits, the qubit technology type (e.g., trapped-ion, superconducting), the commercial launch timeframe, and reference URLs confirming qubit specifications and commercial availability.

For all four entities, each piece of information must be verifiable through the provided reference URLs.
"""


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class CloudInfo(BaseModel):
    provider_name: Optional[str] = None
    market_share_q2_2025: Optional[str] = None  # Keep as string to allow "31%" etc.
    iaas_capability: Optional[str] = None       # e.g., "Yes", "Offers IaaS"
    enterprise_focus: Optional[str] = None      # e.g., "Serves enterprise customers"
    market_share_sources: List[str] = Field(default_factory=list)
    service_sources: List[str] = Field(default_factory=list)


class AcquisitionInfo(BaseModel):
    acquiring_company: Optional[str] = None
    target_company: Optional[str] = None
    deal_value_billion_usd: Optional[str] = None  # e.g., "12.5", "$12.5B"
    payment_structure: Optional[str] = None       # e.g., "all-cash", "stock", "mixed"
    announcement_date: Optional[str] = None       # e.g., "2025-01-12" or "Jan 2025"
    completion_date: Optional[str] = None         # exact completion date
    strategic_purpose: Optional[str] = None
    deal_value_sources: List[str] = Field(default_factory=list)
    timeline_sources: List[str] = Field(default_factory=list)
    strategic_sources: List[str] = Field(default_factory=list)


class SemiFacilityInfo(BaseModel):
    operating_company: Optional[str] = None
    facility_name: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    process_node: Optional[str] = None          # e.g., "4nm", "N4P", "5nm"
    wafer_size: Optional[str] = None            # e.g., "300mm", "12-inch"
    monthly_wafer_capacity: Optional[str] = None  # e.g., "20,000 wpm"
    volume_production_start: Optional[str] = None  # e.g., "Q4 2024", "Dec 2024"
    capacity_timeline: Optional[str] = None       # optional timeline for capacity ramp
    location_sources: List[str] = Field(default_factory=list)
    technology_sources: List[str] = Field(default_factory=list)
    capacity_sources: List[str] = Field(default_factory=list)
    status_sources: List[str] = Field(default_factory=list)
    identity_sources: List[str] = Field(default_factory=list)


class QuantumSystemInfo(BaseModel):
    system_provider: Optional[str] = None
    system_name: Optional[str] = None
    physical_qubits: Optional[str] = None        # keep string (e.g., "127", "100+")
    qubit_technology: Optional[str] = None       # e.g., superconducting, trapped-ion
    launch_timeframe: Optional[str] = None       # e.g., "Q2 2025", "Nov 2025"
    qubit_sources: List[str] = Field(default_factory=list)
    availability_sources: List[str] = Field(default_factory=list)
    identity_sources: List[str] = Field(default_factory=list)


class TechEntitiesExtraction(BaseModel):
    cloud: Optional[CloudInfo] = None
    acquisition: Optional[AcquisitionInfo] = None
    semiconductor_facility: Optional[SemiFacilityInfo] = None
    quantum_system: Optional[QuantumSystemInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all_entities() -> str:
    return """
    Extract structured information for four entities as presented in the answer. Use ONLY information explicitly stated in the answer. If any field is missing, return null for that field. For any "sources" fields, extract and return all URLs explicitly provided in the answer text (including markdown links). Ensure URLs are complete and valid.

    Entity 1 (Cloud Infrastructure Market Leader):
    - provider_name: official name of the cloud provider
    - market_share_q2_2025: the exact market share percentage for Q2 2025 (e.g., "31%")
    - iaas_capability: a short phrase indicating that the provider offers IaaS (e.g., "Offers IaaS", "Yes")
    - enterprise_focus: a short phrase indicating the provider serves enterprise customers (e.g., "Enterprise customers", "Yes")
    - market_share_sources: URLs that directly support Q2 2025 market share and leadership
    - service_sources: URLs that support IaaS offering and enterprise focus

    Entity 2 (Major Technology Acquisition):
    - acquiring_company: name of acquiring company
    - target_company: name of target company
    - deal_value_billion_usd: exact deal value as a number string in billions USD if available (e.g., "12.5" or "$12.5B")
    - payment_structure: "all-cash", "stock", "mixed", or a short description if available
    - announcement_date: date the deal was announced (string as in the answer)
    - completion_date: exact date when the deal closed/completed (string as in the answer)
    - strategic_purpose: a short phrase sentence describing rationale (if provided)
    - deal_value_sources: URLs that confirm the deal value
    - timeline_sources: URLs that confirm the announcement/completion timeline
    - strategic_sources: URLs that discuss sector, acquirer being a major tech firm, or rationale

    Entity 3 (Advanced U.S. Semiconductor Fabrication Facility):
    - operating_company: the company operating the fab
    - facility_name: official site/fab name or designation (if provided)
    - state: U.S. state
    - city: city or metro/region name
    - process_node: specific process node (e.g., "4nm", "5nm", "N4P")
    - wafer_size: wafer size (e.g., "300mm", "12-inch")
    - monthly_wafer_capacity: monthly wafer capacity at full utilization (e.g., "20,000 wpm")
    - volume_production_start: date/timeframe when volume production began
    - capacity_timeline: date/timeframe when stated capacity is/was/will be reached (if available)
    - location_sources: URLs confirming the U.S. location (state + city)
    - technology_sources: URLs confirming process tech and wafer size
    - capacity_sources: URLs confirming capacity metrics
    - status_sources: URLs confirming operational/volume production status
    - identity_sources: URLs confirming operator/facility identity

    Entity 4 (Commercial Quantum Computing System):
    - system_provider: company providing the system
    - system_name: official system/model name
    - physical_qubits: exact number of physical qubits (string, as written in answer)
    - qubit_technology: e.g., "superconducting", "trapped-ion"
    - launch_timeframe: timeframe/date of commercial launch or commercial availability announcement (must be 2025 or later)
    - qubit_sources: URLs confirming qubit count/technology
    - availability_sources: URLs confirming commercial launch/availability
    - identity_sources: URLs confirming system/provider identity
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if isinstance(u, str) and u.strip():
            cleaned.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in cleaned:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def _union_sources(*lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for l in lists:
        merged.extend(_clean_urls(l))
    # Deduplicate
    out: List[str] = []
    seen = set()
    for u in merged:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_cloud_infrastructure_checks(evaluator: Evaluator, parent: VerificationNode, cloud: Optional[CloudInfo]) -> None:
    node_cloud = evaluator.add_parallel(
        id="Cloud_Infrastructure_Market_Leader",
        desc="Identify the cloud infrastructure provider with the highest market share in Q2 2025",
        parent=parent,
        critical=False
    )

    # Unpack with safe defaults
    provider = (cloud.provider_name if cloud else None) or ""
    market_share = (cloud.market_share_q2_2025 if cloud else None) or ""
    ms_sources = _clean_urls(cloud.market_share_sources if cloud else [])
    svc_sources = _clean_urls(cloud.service_sources if cloud else [])
    all_sources = _union_sources(ms_sources, svc_sources)

    # Market Position (critical group)
    market_pos = evaluator.add_parallel(
        id="Market_Position_Q2_2025",
        desc="Verify the provider's market share position in Q2 2025",
        parent=node_cloud,
        critical=True  # All children must be critical
    )
    # Reference presence (critical, gates the two verification leaves)
    ref_market = evaluator.add_custom_node(
        result=len(ms_sources) > 0,
        id="Market_Share_Reference",
        desc="Provide URL reference confirming the market share data for Q2 2025",
        parent=market_pos,
        critical=True
    )
    # Highest market share
    highest_leaf = evaluator.add_leaf(
        id="Highest_Market_Share",
        desc="The provider must have the highest market share percentage among all cloud infrastructure providers in Q2 2025",
        parent=market_pos,
        critical=True
    )
    await evaluator.verify(
        claim=f"In Q2 2025, {provider} held the highest market share in the global cloud infrastructure services market.",
        node=highest_leaf,
        sources=ms_sources,
        additional_instruction="Accept reputable market trackers (e.g., Synergy Research, Canalys, IDC). Confirm that this provider led the overall cloud infrastructure market in Q2 2025."
    )
    # Market share value exact
    ms_value_leaf = evaluator.add_leaf(
        id="Market_Share_Value",
        desc="Provide the specific market share percentage held by the provider in Q2 2025",
        parent=market_pos,
        critical=True
    )
    await evaluator.verify(
        claim=f"In Q2 2025, {provider}'s cloud infrastructure market share was {market_share}.",
        node=ms_value_leaf,
        sources=ms_sources,
        additional_instruction="Allow minor rounding differences (e.g., 31 vs. 31.0%). The page should explicitly list Q2 2025 share for this provider."
    )

    # Service Category (critical group)
    svc_cat = evaluator.add_parallel(
        id="Service_Category",
        desc="Verify the type of cloud services provided",
        parent=node_cloud,
        critical=True
    )
    # Add a sources presence gate for service claims (not in original JSON but added to enforce source-grounding)
    svc_ref = evaluator.add_custom_node(
        result=len(svc_sources) > 0,
        id="Service_Category_Reference",
        desc="Service category claims are backed by at least one URL reference",
        parent=svc_cat,
        critical=True
    )
    # IaaS offering
    iaas_leaf = evaluator.add_leaf(
        id="Infrastructure_as_Service",
        desc="The provider must offer Infrastructure as a Service (IaaS) capabilities",
        parent=svc_cat,
        critical=True
    )
    await evaluator.verify(
        claim=f"{provider} offers Infrastructure as a Service (IaaS).",
        node=iaas_leaf,
        sources=svc_sources,
        additional_instruction="The evidence should indicate IaaS (infrastructure) capabilities (not just SaaS). Enterprise-grade IaaS is acceptable as equivalently phrased."
    )
    # Enterprise focus
    enterprise_leaf = evaluator.add_leaf(
        id="Enterprise_Focus",
        desc="The provider must serve enterprise customers with cloud infrastructure services",
        parent=svc_cat,
        critical=True
    )
    await evaluator.verify(
        claim=f"{provider} serves enterprise customers with its cloud infrastructure services.",
        node=enterprise_leaf,
        sources=svc_sources,
        additional_instruction="Accept statements showing enterprise or business/organization-focused offerings for cloud infrastructure."
    )

    # Provider name (critical leaf under cloud node). Gate it on references above.
    provider_leaf = evaluator.add_leaf(
        id="Provider_Name",
        desc="Provide the official name of the cloud infrastructure provider",
        parent=node_cloud,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official name of the cloud provider is '{provider}'.",
        node=provider_leaf,
        sources=all_sources,
        additional_instruction="Verify that this provider name matches how the company officially styles itself on referenced pages."
    )


async def build_acquisition_checks(evaluator: Evaluator, parent: VerificationNode, acq: Optional[AcquisitionInfo]) -> None:
    node_acq = evaluator.add_parallel(
        id="Major_Technology_Acquisition",
        desc="Identify a major technology acquisition completed in 2025 with deal value exceeding $10 billion",
        parent=parent,
        critical=False
    )

    # Unpack with safe defaults
    acquiring = (acq.acquiring_company if acq else None) or ""
    target = (acq.target_company if acq else None) or ""
    deal_value = (acq.deal_value_billion_usd if acq else None) or ""
    payment_structure = (acq.payment_structure if acq else None) or ""
    ann_date = (acq.announcement_date if acq else None) or ""
    comp_date = (acq.completion_date if acq else None) or ""
    rationale = (acq.strategic_purpose if acq else None) or ""

    deal_sources = _clean_urls(acq.deal_value_sources if acq else [])
    time_sources = _clean_urls(acq.timeline_sources if acq else [])
    strat_sources = _clean_urls(acq.strategic_sources if acq else [])
    core_sources = _union_sources(deal_sources, time_sources, strat_sources)

    # Deal Financial Terms (set non-critical to comply with framework constraint due to non-critical child)
    deal_terms = evaluator.add_parallel(
        id="Deal_Financial_Terms",
        desc="Verify the financial terms of the acquisition",
        parent=node_acq,
        critical=False
    )
    # Reference presence (critical leaf under this group)
    price_ref = evaluator.add_custom_node(
        result=len(deal_sources) > 0,
        id="Deal_Value_Reference",
        desc="Provide URL reference confirming the acquisition deal value",
        parent=deal_terms,
        critical=True
    )
    # Threshold > $10B
    threshold_leaf = evaluator.add_leaf(
        id="Deal_Value_Threshold",
        desc="The acquisition deal value must exceed $10 billion",
        parent=deal_terms,
        critical=True
    )
    await evaluator.verify(
        claim=f"The acquisition of {target} by {acquiring} had a deal value exceeding $10 billion.",
        node=threshold_leaf,
        sources=deal_sources,
        additional_instruction="Confirm the total deal value reported. If multiple figures exist, use the widely cited transaction value and ensure it is > $10B."
    )
    # Exact deal value
    exact_value_leaf = evaluator.add_leaf(
        id="Exact_Deal_Value",
        desc="Provide the specific deal value in billions of dollars",
        parent=deal_terms,
        critical=True
    )
    await evaluator.verify(
        claim=f"The acquisition deal value was {deal_value} billion USD.",
        node=exact_value_leaf,
        sources=deal_sources,
        additional_instruction="Allow common currency rendering (e.g., '$12.5B'). Focus on the numeric magnitude in billions of USD."
    )
    # Payment structure (non-critical)
    if payment_structure.strip():
        pay_struct_leaf = evaluator.add_leaf(
            id="Payment_Structure",
            desc="Specify whether the deal was all-cash, stock, or mixed consideration",
            parent=deal_terms,
            critical=False
        )
        await evaluator.verify(
            claim=f"The consideration structure for this deal was '{payment_structure}'.",
            node=pay_struct_leaf,
            sources=deal_sources,
            additional_instruction="Accept phrasing that indicates all-cash, stock-for-stock, cash-and-stock, or equivalent structure."
        )

    # Transaction Timeline (set non-critical due to non-critical child 'Announcement_Date')
    timeline = evaluator.add_parallel(
        id="Transaction_Timeline",
        desc="Verify the timeline of the acquisition transaction",
        parent=node_acq,
        critical=False
    )
    time_ref = evaluator.add_custom_node(
        result=len(time_sources) > 0,
        id="Timeline_Reference",
        desc="Provide URL reference confirming the acquisition timeline",
        parent=timeline,
        critical=True
    )
    # Completion year constraint (critical)
    completion_year_leaf = evaluator.add_leaf(
        id="Completion_Year",
        desc="The acquisition must have been completed in 2025 or early 2026",
        parent=timeline,
        critical=True
    )
    await evaluator.verify(
        claim=f"The acquisition of {target} by {acquiring} was completed in 2025 or early 2026 (e.g., by March 2026).",
        node=completion_year_leaf,
        sources=time_sources,
        additional_instruction="Confirm the closing date on authoritative sources. 'Early 2026' should be reasonably interpreted as Q1 2026."
    )
    # Completion exact date (critical)
    completion_date_leaf = evaluator.add_leaf(
        id="Completion_Date",
        desc="Provide the specific date when the acquisition was completed",
        parent=timeline,
        critical=True
    )
    await evaluator.verify(
        claim=f"The acquisition closed/completed on {comp_date}.",
        node=completion_date_leaf,
        sources=time_sources,
        additional_instruction="The page should explicitly state the closing or completion date."
    )
    # Announcement date (non-critical)
    if ann_date.strip():
        announcement_leaf = evaluator.add_leaf(
            id="Announcement_Date",
            desc="Provide the date when the acquisition was announced",
            parent=timeline,
            critical=False
        )
        await evaluator.verify(
            claim=f"The acquisition was announced on {ann_date}.",
            node=announcement_leaf,
            sources=time_sources,
            additional_instruction="Verify the announcement date if provided."
        )

    # Strategic Nature (set non-critical to allow optional rationale)
    strategy = evaluator.add_parallel(
        id="Strategic_Nature",
        desc="Verify the strategic characteristics of the acquisition",
        parent=node_acq,
        critical=False
    )
    # Technology sector (critical)
    tech_sector_leaf = evaluator.add_leaf(
        id="Technology_Sector",
        desc="The acquisition must be in the technology sector",
        parent=strategy,
        critical=True
    )
    await evaluator.verify(
        claim=f"The acquisition of {target} by {acquiring} is in the technology sector.",
        node=tech_sector_leaf,
        sources=core_sources,
        additional_instruction="Confirm that the target/acquirer operate within the technology sector."
    )
    # Acquirer type (critical)
    acquirer_type_leaf = evaluator.add_leaf(
        id="Acquirer_Type",
        desc="The acquiring company must be a major technology company",
        parent=strategy,
        critical=True
    )
    await evaluator.verify(
        claim=f"{acquiring} is a major technology company.",
        node=acquirer_type_leaf,
        sources=core_sources,
        additional_instruction="Accept well-recognized leading tech firms; confirmation can come from company profiles, press releases, or reputable news."
    )
    # Strategic purpose (non-critical)
    if rationale.strip():
        strat_purpose_leaf = evaluator.add_leaf(
            id="Strategic_Purpose",
            desc="Provide the stated strategic purpose or rationale for the acquisition",
            parent=strategy,
            critical=False
        )
        await evaluator.verify(
            claim=f"The stated strategic purpose/rationale for this acquisition was: {rationale}",
            node=strat_purpose_leaf,
            sources=core_sources,
            additional_instruction="Look for quotes or paraphrased reasons in press releases, investor materials, or reliable news."
        )

    # Parties (critical)
    parties = evaluator.add_parallel(
        id="Acquisition_Parties",
        desc="Identify the acquiring company and target company",
        parent=node_acq,
        critical=True
    )
    parties_sources = _union_sources(deal_sources, time_sources, strat_sources)
    acquirer_leaf = evaluator.add_leaf(
        id="Acquiring_Company",
        desc="Provide the name of the acquiring company",
        parent=parties,
        critical=True
    )
    await evaluator.verify(
        claim=f"The acquiring company was {acquiring}.",
        node=acquirer_leaf,
        sources=parties_sources,
        additional_instruction="The page should clearly identify the acquirer."
    )
    target_leaf = evaluator.add_leaf(
        id="Target_Company",
        desc="Provide the name of the acquired/target company",
        parent=parties,
        critical=True
    )
    await evaluator.verify(
        claim=f"The target company was {target}.",
        node=target_leaf,
        sources=parties_sources,
        additional_instruction="The page should clearly identify the acquired or target company."
    )


async def build_semiconductor_facility_checks(evaluator: Evaluator, parent: VerificationNode, fab: Optional[SemiFacilityInfo]) -> None:
    node_fab = evaluator.add_parallel(
        id="Advanced_Semiconductor_Facility",
        desc="Identify an advanced semiconductor fabrication facility in the United States with specific technical capabilities",
        parent=parent,
        critical=False
    )

    # Unpack
    operator = (fab.operating_company if fab else None) or ""
    fac_name = (fab.facility_name if fab else None) or ""
    state = (fab.state if fab else None) or ""
    city = (fab.city if fab else None) or ""
    process_node = (fab.process_node if fab else None) or ""
    wafer_size = (fab.wafer_size if fab else None) or ""
    capacity = (fab.monthly_wafer_capacity if fab else None) or ""
    vp_start = (fab.volume_production_start if fab else None) or ""
    cap_timeline = (fab.capacity_timeline if fab else None) or ""

    loc_sources = _clean_urls(fab.location_sources if fab else [])
    tech_sources = _clean_urls(fab.technology_sources if fab else [])
    cap_sources = _clean_urls(fab.capacity_sources if fab else [])
    status_sources = _clean_urls(fab.status_sources if fab else [])
    id_sources = _clean_urls(fab.identity_sources if fab else [])
    all_sources = _union_sources(loc_sources, tech_sources, cap_sources, status_sources, id_sources)

    # Geographic Requirements (critical)
    geo = evaluator.add_parallel(
        id="Geographic_Requirements",
        desc="Verify the geographic location of the semiconductor facility",
        parent=node_fab,
        critical=True
    )
    loc_ref = evaluator.add_custom_node(
        result=len(loc_sources) > 0,
        id="Location_Reference",
        desc="Provide URL reference confirming the facility location",
        parent=geo,
        critical=True
    )
    us_leaf = evaluator.add_leaf(
        id="US_Location",
        desc="The facility must be located in the United States",
        parent=geo,
        critical=True
    )
    await evaluator.verify(
        claim=f"The fabrication facility operated by {operator} is located in the United States.",
        node=us_leaf,
        sources=loc_sources,
        additional_instruction="The source should explicitly indicate a U.S. location."
    )
    state_leaf = evaluator.add_leaf(
        id="Specific_State",
        desc="Provide the specific U.S. state where the facility is located",
        parent=geo,
        critical=True
    )
    await evaluator.verify(
        claim=f"The facility is located in the U.S. state of {state}.",
        node=state_leaf,
        sources=loc_sources,
        additional_instruction="Verify the specific state location."
    )
    city_leaf = evaluator.add_leaf(
        id="City_or_Region",
        desc="Provide the specific city or metropolitan area where the facility is located",
        parent=geo,
        critical=True
    )
    await evaluator.verify(
        claim=f"The facility is located in {city}, {state}.",
        node=city_leaf,
        sources=loc_sources,
        additional_instruction="Allow equivalent metro/region naming if commonly used."
    )

    # Technical Specifications (critical)
    tech = evaluator.add_parallel(
        id="Technical_Specifications",
        desc="Verify the technical capabilities and specifications of the facility",
        parent=node_fab,
        critical=True
    )
    tech_ref = evaluator.add_custom_node(
        result=len(tech_sources) > 0,
        id="Technology_Reference",
        desc="Provide URL reference confirming the process technology specifications",
        parent=tech,
        critical=True
    )
    proc_leaf = evaluator.add_leaf(
        id="Process_Technology",
        desc="The facility must use advanced process node technology (10nm or below)",
        parent=tech,
        critical=True
    )
    await evaluator.verify(
        claim=f"The facility uses advanced process node technology of 10nm or below.",
        node=proc_leaf,
        sources=tech_sources,
        additional_instruction="Accept synonyms like 'N5/N4/N3' indicating ≤10nm class processes."
    )
    specific_node_leaf = evaluator.add_leaf(
        id="Specific_Process_Node",
        desc="Provide the specific process node technology used (e.g., 4nm, 5nm, 7nm)",
        parent=tech,
        critical=True
    )
    await evaluator.verify(
        claim=f"The specific process node used at this facility is {process_node}.",
        node=specific_node_leaf,
        sources=tech_sources,
        additional_instruction="Allow process node naming conventions like 'N4P', 'Intel 4', etc., mapping to their respective nm-class."
    )
    wafer_leaf = evaluator.add_leaf(
        id="Wafer_Size",
        desc="Provide the wafer size used in production (e.g., 12-inch, 300mm)",
        parent=tech,
        critical=True
    )
    await evaluator.verify(
        claim=f"The wafer size used in production at the facility is {wafer_size}.",
        node=wafer_leaf,
        sources=tech_sources,
        additional_instruction="Treat 300mm and 12-inch as equivalent."
    )

    # Production Capacity (set non-critical due to optional timeline subnode)
    capacity_node = evaluator.add_parallel(
        id="Production_Capacity",
        desc="Verify the production capacity metrics of the facility",
        parent=node_fab,
        critical=False
    )
    cap_ref = evaluator.add_custom_node(
        result=len(cap_sources) > 0,
        id="Capacity_Reference",
        desc="Provide URL reference confirming the production capacity",
        parent=capacity_node,
        critical=True
    )
    capacity_leaf = evaluator.add_leaf(
        id="Monthly_Wafer_Capacity",
        desc="Provide the monthly wafer production capacity at full utilization",
        parent=capacity_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The monthly wafer production capacity at full utilization is {capacity}.",
        node=capacity_leaf,
        sources=cap_sources,
        additional_instruction="Accept units like 'wpm' (wafers per month). The number should match or be very close."
    )
    if cap_timeline.strip():
        cap_timeline_leaf = evaluator.add_leaf(
            id="Capacity_Timeline",
            desc="Specify when the facility reached or will reach stated capacity",
            parent=capacity_node,
            critical=False
        )
        await evaluator.verify(
            claim=f"The facility's capacity timeline indicates: {cap_timeline}.",
            node=cap_timeline_leaf,
            sources=cap_sources,
            additional_instruction="Verify any stated date/timeframe for reaching capacity."
        )

    # Operational Status (critical)
    status_node = evaluator.add_parallel(
        id="Operational_Status",
        desc="Verify the current operational status of the facility",
        parent=node_fab,
        critical=True
    )
    status_ref = evaluator.add_custom_node(
        result=len(status_sources) > 0,
        id="Status_Reference",
        desc="Provide URL reference confirming the operational status",
        parent=status_node,
        critical=True
    )
    prod_status_leaf = evaluator.add_leaf(
        id="Production_Status",
        desc="The facility must be in volume production or have entered volume production by end of 2024",
        parent=status_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The facility entered volume production by the end of 2024 (or earlier).",
        node=prod_status_leaf,
        sources=status_sources,
        additional_instruction="If volume production started any time up to and including Dec 2024 (or earlier), this passes."
    )
    status_timeline_leaf = evaluator.add_leaf(
        id="Status_Timeline",
        desc="Provide the date or timeframe when volume production began",
        parent=status_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Volume production began in {vp_start}.",
        node=status_timeline_leaf,
        sources=status_sources,
        additional_instruction="The source should state the volume production start date or timeframe."
    )

    # Facility Identity (set non-critical to allow optional facility name)
    identity_node = evaluator.add_parallel(
        id="Facility_Identity",
        desc="Identify the facility operator and official name",
        parent=node_fab,
        critical=False
    )
    identity_sources_all = _union_sources(id_sources, all_sources)
    operator_leaf = evaluator.add_leaf(
        id="Operating_Company",
        desc="Provide the name of the company operating the facility",
        parent=identity_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The facility is operated by {operator}.",
        node=operator_leaf,
        sources=identity_sources_all,
        additional_instruction="The operator should be clearly identified on at least one cited page."
    )
    if fac_name.strip():
        facility_name_leaf = evaluator.add_leaf(
            id="Facility_Name",
            desc="Provide the official name or designation of the facility",
            parent=identity_node,
            critical=False
        )
        await evaluator.verify(
            claim=f"The official name/designation of the facility is '{fac_name}'.",
            node=facility_name_leaf,
            sources=identity_sources_all,
            additional_instruction="Accept commonly used official fab/site designations."
        )


async def build_quantum_system_checks(evaluator: Evaluator, parent: VerificationNode, qsys: Optional[QuantumSystemInfo]) -> None:
    node_q = evaluator.add_parallel(
        id="Commercial_Quantum_Computing_System",
        desc="Identify a commercial quantum computing system with specific qubit count and performance characteristics",
        parent=parent,
        critical=False
    )

    # Unpack
    provider = (qsys.system_provider if qsys else None) or ""
    system_name = (qsys.system_name if qsys else None) or ""
    qubits = (qsys.physical_qubits if qsys else None) or ""
    qtech = (qsys.qubit_technology if qsys else None) or ""
    launch_tf = (qsys.launch_timeframe if qsys else None) or ""

    q_sources = _clean_urls(qsys.qubit_sources if qsys else [])
    a_sources = _clean_urls(qsys.availability_sources if qsys else [])
    id_sources = _clean_urls(qsys.identity_sources if qsys else [])
    all_sources = _union_sources(q_sources, a_sources, id_sources)

    # Qubit Specifications (critical)
    qubit_specs = evaluator.add_parallel(
        id="Qubit_Specifications",
        desc="Verify the qubit count and type of the quantum system",
        parent=node_q,
        critical=True
    )
    qubit_ref = evaluator.add_custom_node(
        result=len(q_sources) > 0,
        id="Qubit_Reference",
        desc="Provide URL reference confirming the qubit specifications",
        parent=qubit_specs,
        critical=True
    )
    # At least 90 physical qubits
    min_qubits_leaf = evaluator.add_leaf(
        id="Physical_Qubit_Count",
        desc="The system must have at least 90 physical qubits",
        parent=qubit_specs,
        critical=True
    )
    await evaluator.verify(
        claim=f"The system '{system_name}' from {provider} has at least 90 physical qubits.",
        node=min_qubits_leaf,
        sources=q_sources,
        additional_instruction="Ensure this refers to physical (not logical) qubits."
    )
    # Exact number
    exact_qubits_leaf = evaluator.add_leaf(
        id="Exact_Qubit_Number",
        desc="Provide the exact number of physical qubits in the system",
        parent=qubit_specs,
        critical=True
    )
    await evaluator.verify(
        claim=f"The system '{system_name}' has {qubits} physical qubits.",
        node=exact_qubits_leaf,
        sources=q_sources,
        additional_instruction="If multiple versions exist, verify the version specified in the answer."
    )
    # Qubit technology type
    tech_type_leaf = evaluator.add_leaf(
        id="Qubit_Technology",
        desc="Specify the qubit technology type (e.g., trapped-ion, superconducting, etc.)",
        parent=qubit_specs,
        critical=True
    )
    await evaluator.verify(
        claim=f"The qubit technology used is {qtech}.",
        node=tech_type_leaf,
        sources=q_sources,
        additional_instruction="Accept common families (superconducting, trapped-ion, neutral atom, photonic, spin, etc.)."
    )

    # Commercial Availability (critical)
    availability = evaluator.add_parallel(
        id="Commercial_Availability",
        desc="Verify the commercial availability status of the system",
        parent=node_q,
        critical=True
    )
    avail_ref = evaluator.add_custom_node(
        result=len(a_sources) > 0,
        id="Availability_Reference",
        desc="Provide URL reference confirming commercial availability",
        parent=availability,
        critical=True
    )
    # Commercial launch/announcement
    launch_leaf = evaluator.add_leaf(
        id="Commercial_Launch",
        desc="The system must have been commercially launched or announced for commercial availability",
        parent=availability,
        critical=True
    )
    await evaluator.verify(
        claim=f"The system '{system_name}' was commercially launched or announced for commercial availability.",
        node=launch_leaf,
        sources=a_sources,
        additional_instruction="The page should indicate commercial launch or explicit commercial availability announcement."
    )
    # Launch timeframe (must be 2025 or later)
    timeframe_leaf = evaluator.add_leaf(
        id="Launch_Timeframe",
        desc="Provide the timeframe of commercial launch or announcement (must be 2025 or later)",
        parent=availability,
        critical=True
    )
    await evaluator.verify(
        claim=f"The commercial launch or availability announcement for '{system_name}' occurred in 2025 or later (timeframe stated as: {launch_tf}).",
        node=timeframe_leaf,
        sources=a_sources,
        additional_instruction="Confirm that the event happened in 2025 or later."
    )

    # System Identity (critical)
    identity = evaluator.add_parallel(
        id="System_Identity",
        desc="Identify the quantum computing system and provider",
        parent=node_q,
        critical=True
    )
    provider_leaf = evaluator.add_leaf(
        id="System_Provider",
        desc="Provide the name of the company or organization providing the quantum system",
        parent=identity,
        critical=True
    )
    await evaluator.verify(
        claim=f"The system provider is {provider}.",
        node=provider_leaf,
        sources=all_sources,
        additional_instruction="Verify the company's name as the provider of this system."
    )
    name_leaf = evaluator.add_leaf(
        id="System_Name",
        desc="Provide the official name or model designation of the quantum system",
        parent=identity,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official name/model designation of the quantum computing system is '{system_name}'.",
        node=name_leaf,
        sources=all_sources,
        additional_instruction="Verify the model/system name exactly or with minor permissible variations."
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator and root (parallel aggregation across 4 entities)
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

    # Extract all entities from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all_entities(),
        template_class=TechEntitiesExtraction,
        extraction_name="tech_entities_extraction",
    )

    # Build sub-trees for each entity
    await build_cloud_infrastructure_checks(evaluator, root, extracted.cloud if extracted else None)
    await build_acquisition_checks(evaluator, root, extracted.acquisition if extracted else None)
    await build_semiconductor_facility_checks(evaluator, root, extracted.semiconductor_facility if extracted else None)
    await build_quantum_system_checks(evaluator, root, extracted.quantum_system if extracted else None)

    # Return evaluation summary
    return evaluator.get_summary()