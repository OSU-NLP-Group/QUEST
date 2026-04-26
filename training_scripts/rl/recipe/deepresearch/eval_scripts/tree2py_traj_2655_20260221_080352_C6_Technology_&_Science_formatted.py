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
TASK_ID = "multi_state_colocation"
TASK_DESCRIPTION = (
    "A financial services enterprise needs to establish a geographically distributed colocation infrastructure "
    "consisting of two facilities: a primary production facility and a disaster recovery (DR) backup facility. "
    "Both facilities must meet the following requirements:\n\n"
    "Tier and Availability Requirements:\n"
    "- Hold Uptime Institute Tier III certification or higher (Tier III for concurrent maintainability, or Tier IV for fault tolerance)\n"
    "- Guarantee minimum 99.99% uptime SLA\n\n"
    "Compliance Requirements:\n"
    "- Maintain valid SOC 2 Type II certification\n"
    "- Maintain valid ISO 27001 certification\n"
    "- Maintain valid PCI DSS certification\n\n"
    "Technical Requirements:\n"
    "- Support minimum 500 kW of deployable power capacity for wholesale colocation\n"
    "- Be carrier-neutral with access to at least 10 network service providers\n"
    "- Provide cross-connect services for direct interconnection to network providers\n\n"
    "Geographic Requirements:\n"
    "- The primary and DR facilities must be located in different US states to ensure geographic redundancy and avoid common regional disaster scenarios\n\n"
    "Identify specific colocation facilities (including facility name, address, and provider) for both the primary and DR locations that meet all of these requirements. "
    "For each facility, provide reference URLs documenting: (1) tier certification status, (2) compliance certifications, and "
    "(3) technical specifications including power capacity and network connectivity options."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilityInfo(BaseModel):
    # Core identification
    provider: Optional[str] = None
    facility_name: Optional[str] = None
    address: Optional[str] = None
    state: Optional[str] = None

    # Tier & availability
    tier_level: Optional[str] = None  # e.g., "Tier III", "Tier IV"
    uptime_sla: Optional[str] = None  # e.g., "99.99%"

    # Compliance
    soc2_type_ii: Optional[str] = None
    iso27001: Optional[str] = None
    pci_dss: Optional[str] = None

    # Technical specs
    power_capacity_kw: Optional[str] = None   # Keep as string like "500 kW", ">= 0.5 MW"
    carrier_neutral: Optional[str] = None     # e.g., "carrier-neutral", "yes", "true"
    num_network_providers: Optional[str] = None  # e.g., "10+", "at least 15"
    cross_connect: Optional[str] = None       # e.g., "cross-connects available"

    # Reference URLs (explicitly present in answer)
    tier_urls: List[str] = Field(default_factory=list)          # For tier certification evidence
    uptime_urls: List[str] = Field(default_factory=list)        # For uptime SLA evidence
    compliance_urls: List[str] = Field(default_factory=list)    # For SOC2, ISO27001, PCI DSS evidence
    specs_urls: List[str] = Field(default_factory=list)         # For power capacity specs
    connectivity_urls: List[str] = Field(default_factory=list)  # For carrier-neutrality & cross-connect evidence
    general_urls: List[str] = Field(default_factory=list)       # Provider/facility pages with address, name, etc.


class DeploymentExtraction(BaseModel):
    primary: Optional[FacilityInfo] = None
    dr: Optional[FacilityInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_deployment() -> str:
    return (
        "Extract exactly two colocation facilities from the answer: one 'primary' production facility and one 'dr' backup facility. "
        "For each facility, extract the following fields (return null if a field is not explicitly provided in the answer text):\n"
        "- provider: Company operating the colocation facility.\n"
        "- facility_name: Specific data center/facility name.\n"
        "- address: Street address or full address string.\n"
        "- state: The US state abbreviation or full name for the facility location.\n"
        "- tier_level: The Uptime Institute tier level mentioned (e.g., 'Tier III', 'Tier IV').\n"
        "- uptime_sla: The uptime SLA percentage mentioned (e.g., '99.99%').\n"
        "- soc2_type_ii: A string indicating SOC 2 Type II certification if explicitly mentioned.\n"
        "- iso27001: A string indicating ISO 27001 certification if explicitly mentioned.\n"
        "- pci_dss: A string indicating PCI DSS certification if explicitly mentioned.\n"
        "- power_capacity_kw: A string indicating the deployable power capacity (e.g., '500 kW', '0.5 MW', '>= 500 kW').\n"
        "- carrier_neutral: A string indicating carrier-neutral status if explicitly mentioned.\n"
        "- num_network_providers: A string indicating the number of network service providers available (e.g., '10+', 'at least 10').\n"
        "- cross_connect: A string indicating cross-connect availability if explicitly mentioned.\n"
        "Also extract explicit reference URLs present in the answer for each evidence category:\n"
        "- tier_urls: URLs documenting tier certification status for the specific facility.\n"
        "- uptime_urls: URLs documenting uptime SLA (if provided).\n"
        "- compliance_urls: URLs documenting compliance certifications (SOC 2 Type II, ISO 27001, PCI DSS).\n"
        "- specs_urls: URLs documenting technical specifications, especially power capacity.\n"
        "- connectivity_urls: URLs documenting network connectivity options (carrier-neutral, provider counts, cross-connects).\n"
        "- general_urls: URLs that identify the specific facility, address, provider.\n\n"
        "Return a JSON object with two top-level fields: 'primary' and 'dr', each an object with all fields above. "
        "If the answer provides more than two facilities, choose the most clearly labeled as 'primary' and 'dr'. "
        "If labels are not explicit, select the first two suitable facilities mentioned in the answer as 'primary' then 'dr'. "
        "All URLs must be explicitly present in the answer (including markdown links). Do not invent URLs."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        key = u.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _select_urls(*groups: List[str]) -> List[str]:
    # Merge and deduplicate URL groups
    merged: List[str] = []
    for g in groups:
        merged.extend(g or [])
    return _dedup_urls(merged)


def _safe(val: Optional[str], default: str) -> str:
    return val.strip() if isinstance(val, str) and val.strip() else default


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_facility(
    evaluator: Evaluator,
    parent_node,
    fac: FacilityInfo,
    label_prefix: str,
) -> None:
    """
    Build verification sub-tree for a facility (Primary or DR).
    All children are marked critical to satisfy root critical constraints.
    """

    # Section: Tier & Uptime
    tier_uptime_node = evaluator.add_parallel(
        id=f"{label_prefix}_Tier_and_Uptime",
        desc=f"{label_prefix} facility must have appropriate tier certification and uptime SLA guarantees",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Tier certification (Tier III or higher)
    tier_claim = (
        "This facility holds an Uptime Institute Tier III or Tier IV certification (i.e., Tier III or higher)."
    )
    tier_cert_node = evaluator.add_leaf(
        id=f"{label_prefix}_Tier_Certification",
        desc=f"{label_prefix}: Facility must hold Uptime Institute Tier III or higher certification",
        parent=tier_uptime_node,
        critical=True,
    )
    tier_sources = _select_urls(fac.tier_urls, fac.general_urls)
    await evaluator.verify(
        claim=tier_claim,
        node=tier_cert_node,
        sources=tier_sources,
        additional_instruction=(
            "Verify explicitly from the provided URLs whether the specific facility lists an Uptime Institute Tier certification "
            "at level 3 (Tier III) or level 4 (Tier IV). Accept equivalent phrases like 'Tier 3', 'Tier IV', or "
            "'Uptime Institute Tier III' wording on the facility's page or official certifications listing."
        ),
    )

    # Leaf: Uptime SLA >= 99.99%
    uptime_claim = (
        "The provider guarantees an uptime SLA of at least 99.99% for this facility or its hosting service."
    )
    uptime_node = evaluator.add_leaf(
        id=f"{label_prefix}_Uptime_SLA",
        desc=f"{label_prefix}: Provider must guarantee minimum 99.99% uptime SLA",
        parent=tier_uptime_node,
        critical=True,
    )
    uptime_sources = _select_urls(fac.uptime_urls, fac.general_urls, fac.specs_urls, fac.tier_urls)
    await evaluator.verify(
        claim=uptime_claim,
        node=uptime_node,
        sources=uptime_sources,
        additional_instruction=(
            "Look for SLA documentation or provider pages stating uptime commitments. Accept equivalent language indicating "
            "≥99.99% availability. If multiple SLA levels are shown, ensure at least one applicable service tier meets 99.99%."
        ),
    )

    # Leaf: Tier reference URL(s) existence (custom)
    evaluator.add_custom_node(
        result=bool(fac.tier_urls),
        id=f"{label_prefix}_Tier_Reference",
        desc=f"{label_prefix}: Provide reference URL documenting the facility's tier certification status",
        parent=tier_uptime_node,
        critical=True,
    )

    # Section: Compliance Certifications
    compliance_node = evaluator.add_parallel(
        id=f"{label_prefix}_Compliance_Certifications",
        desc=f"{label_prefix} facility must hold all required compliance certifications for financial services workloads",
        parent=parent_node,
        critical=True,
    )

    # Leaf: SOC 2 Type II
    soc2_claim = "This facility or provider maintains a valid SOC 2 Type II certification."
    soc2_node = evaluator.add_leaf(
        id=f"{label_prefix}_SOC2_Certification",
        desc=f"{label_prefix}: Facility must maintain valid SOC 2 Type II certification",
        parent=compliance_node,
        critical=True,
    )
    comp_sources = _select_urls(fac.compliance_urls, fac.general_urls)
    await evaluator.verify(
        claim=soc2_claim,
        node=soc2_node,
        sources=comp_sources,
        additional_instruction=(
            "Confirm from certification listings or provider compliance pages that SOC 2 Type II is current/valid. "
            "Accept evidence referencing SOC 2 Type II attestations or audit reports."
        ),
    )

    # Leaf: ISO 27001
    iso_claim = "This facility or provider maintains a valid ISO 27001 certification for information security management."
    iso_node = evaluator.add_leaf(
        id=f"{label_prefix}_ISO27001_Certification",
        desc=f"{label_prefix}: Facility must maintain valid ISO 27001 certification",
        parent=compliance_node,
        critical=True,
    )
    await evaluator.verify(
        claim=iso_claim,
        node=iso_node,
        sources=comp_sources,
        additional_instruction=(
            "Confirm from certification listings or provider security/compliance pages that ISO 27001 certification is held "
            "and applicable to the facility or service organization."
        ),
    )

    # Leaf: PCI DSS
    pci_claim = "This facility or provider maintains a valid PCI DSS certification applicable to colocation or related services."
    pci_node = evaluator.add_leaf(
        id=f"{label_prefix}_PCI_DSS_Certification",
        desc=f"{label_prefix}: Facility must maintain valid PCI DSS certification",
        parent=compliance_node,
        critical=True,
    )
    await evaluator.verify(
        claim=pci_claim,
        node=pci_node,
        sources=comp_sources,
        additional_instruction=(
            "Confirm from provider compliance pages or attestations that PCI DSS certification is applicable "
            "to services used in the colocation facility."
        ),
    )

    # Leaf: Compliance reference URLs existence (custom)
    evaluator.add_custom_node(
        result=bool(fac.compliance_urls),
        id=f"{label_prefix}_Compliance_Reference",
        desc=f"{label_prefix}: Provide reference URL documenting the facility's compliance certifications",
        parent=compliance_node,
        critical=True,
    )

    # Section: Technical Specifications
    tech_node = evaluator.add_parallel(
        id=f"{label_prefix}_Technical_Specifications",
        desc=f"{label_prefix} facility must meet minimum power capacity and network connectivity requirements",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Power capacity >= 500 kW
    power_claim = (
        "This facility supports at least 500 kW of deployable power capacity for wholesale colocation."
    )
    power_node = evaluator.add_leaf(
        id=f"{label_prefix}_Power_Capacity",
        desc=f"{label_prefix}: Facility must support minimum 500 kW of deployable power capacity for wholesale colocation",
        parent=tech_node,
        critical=True,
    )
    power_sources = _select_urls(fac.specs_urls, fac.general_urls)
    await evaluator.verify(
        claim=power_claim,
        node=power_node,
        sources=power_sources,
        additional_instruction=(
            "Verify power capacity on technical specification or facility datasheet pages. Accept equivalent figures (e.g., "
            "0.5 MW) indicating ≥500 kW available for deployment."
        ),
    )

    # Leaf: Carrier-neutral with ≥10 network providers
    carriers_claim = (
        "This facility is carrier-neutral and has access to at least 10 network service providers."
    )
    carriers_node = evaluator.add_leaf(
        id=f"{label_prefix}_Carrier_Neutrality",
        desc=f"{label_prefix}: Facility must be carrier-neutral with access to at least 10 network service providers",
        parent=tech_node,
        critical=True,
    )
    connectivity_sources = _select_urls(fac.connectivity_urls, fac.specs_urls, fac.general_urls)
    await evaluator.verify(
        claim=carriers_claim,
        node=carriers_node,
        sources=connectivity_sources,
        additional_instruction=(
            "Check facility or provider network/peering/carrier lists. Accept evidence indicating 'carrier-neutral' and listing "
            "ten or more carriers/providers, or phrasing like '10+' or 'at least ten'."
        ),
    )

    # Leaf: Cross-connect services available
    xconnect_claim = (
        "This facility provides cross-connect services for direct interconnection to network providers."
    )
    xconnect_node = evaluator.add_leaf(
        id=f"{label_prefix}_Cross_Connect",
        desc=f"{label_prefix}: Facility must provide cross-connect services for direct interconnection to network providers",
        parent=tech_node,
        critical=True,
    )
    await evaluator.verify(
        claim=xconnect_claim,
        node=xconnect_node,
        sources=connectivity_sources,
        additional_instruction=(
            "Confirm references to cross-connect offerings (e.g., fiber/copper cross-connects) enabling direct interconnection "
            "to carriers or providers."
        ),
    )

    # Leaf: Technical specifications reference URLs existence (custom)
    evaluator.add_custom_node(
        result=bool(fac.specs_urls) or bool(fac.connectivity_urls),
        id=f"{label_prefix}_Specifications_Reference",
        desc=f"{label_prefix}: Provide reference URL documenting the facility's technical specifications including power capacity and network connectivity options",
        parent=tech_node,
        critical=True,
    )

    # Section: Location Requirements
    location_node = evaluator.add_parallel(
        id=f"{label_prefix}_Location_Requirements",
        desc=f"{label_prefix} facility identification including state location, facility name, address, and provider",
        parent=parent_node,
        critical=True,
    )

    # Leaf: State location verification
    stated_state = _safe(fac.state, "unknown state")
    state_claim = (
        f"The facility is located in the US state of {stated_state}."
    )
    state_node = evaluator.add_leaf(
        id=f"{label_prefix}_State_Location",
        desc=f"{label_prefix}: Identify the specific US state where the facility is located",
        parent=location_node,
        critical=True,
    )
    state_sources = _select_urls(fac.general_urls, fac.specs_urls)
    await evaluator.verify(
        claim=state_claim,
        node=state_node,
        sources=state_sources,
        additional_instruction=(
            "Verify the facility's address or location page indicates a city and state that match the stated US state."
        ),
    )

    # Leaf: Specific facility identity (name, address, provider)
    fac_name = _safe(fac.facility_name, "the selected facility")
    fac_addr = _safe(fac.address, "the stated address")
    fac_prov = _safe(fac.provider, "the stated provider")
    identity_claim = (
        f"The facility named '{fac_name}' at address '{fac_addr}' is operated by '{fac_prov}'."
    )
    identity_node = evaluator.add_leaf(
        id=f"{label_prefix}_Specific_Facility",
        desc=f"{label_prefix}: Identify the specific facility name, address, and provider meeting all requirements",
        parent=location_node,
        critical=True,
    )
    identity_sources = _select_urls(fac.general_urls, fac.specs_urls, fac.connectivity_urls)
    await evaluator.verify(
        claim=identity_claim,
        node=identity_node,
        sources=identity_sources,
        additional_instruction=(
            "Confirm on the provider or facility page that the facility name, operator (provider/company), and physical address "
            "match the stated details."
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Multi-State Enterprise Colocation Deployment task.
    """

    # Initialize evaluator - root is critical with parallel aggregation
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel: primary & DR & state check evaluated independently
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

    # IMPORTANT: The root node in the rubric is critical; to satisfy framework constraints,
    # all children of a critical node must also be critical.
    # Therefore, we mark all immediate children under root as critical=True.

    # Extract deployment details
    extraction = await evaluator.extract(
        prompt=prompt_extract_deployment(),
        template_class=DeploymentExtraction,
        extraction_name="deployment_extraction",
    )

    # Build facility subtrees
    # Primary
    primary_parent = evaluator.add_parallel(
        id="Primary_Colocation_Facility",
        desc="Primary production colocation facility meeting all enterprise requirements for tier certification, compliance, power, and connectivity",
        parent=root,
        critical=True,
    )
    if extraction.primary is None:
        extraction.primary = FacilityInfo()
    await verify_facility(evaluator, primary_parent, extraction.primary, "Primary")

    # DR
    dr_parent = evaluator.add_parallel(
        id="DR_Backup_Colocation_Facility",
        desc="Disaster recovery backup colocation facility meeting equivalent technical and compliance standards as primary facility",
        parent=root,
        critical=True,
    )
    if extraction.dr is None:
        extraction.dr = FacilityInfo()
    await verify_facility(evaluator, dr_parent, extraction.dr, "DR")

    # Different State Requirement (logical check)
    diff_state_node = evaluator.add_leaf(
        id="Different_State_Requirement",
        desc="Primary and DR facilities must be located in different US states to ensure geographic redundancy and avoid common regional disaster scenarios",
        parent=root,
        critical=True,
    )
    primary_state = _safe(extraction.primary.state, "unknown")
    dr_state = _safe(extraction.dr.state, "unknown")
    diff_claim = (
        f"The primary facility state '{primary_state}' and the DR facility state '{dr_state}' are different US states."
    )
    await evaluator.verify(
        claim=diff_claim,
        node=diff_state_node,
        additional_instruction=(
            "This is a simple logical check based on the extracted states. "
            "Return Correct if the two state strings are clearly different US states; "
            "Return Incorrect if they are the same or either is unknown/unparseable."
        ),
    )

    # Return structured summary
    return evaluator.get_summary()