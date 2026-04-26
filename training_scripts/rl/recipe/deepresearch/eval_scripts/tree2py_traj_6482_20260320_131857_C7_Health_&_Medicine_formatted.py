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
TASK_ID = "nh_vax_2022_state_requirements"
TASK_DESCRIPTION = """
According to published analyses from 2022, four U.S. states achieved nursing home staff vaccination rates of 99%, placing them among the highest in the nation. Identify ONE of these four states and provide comprehensive documentation of all the specific vaccination and regulatory compliance requirements that a Medicare/Medicaid-certified nursing home in that state must meet. Your answer must include detailed information about each of the following 14 requirement categories: (1) The state's specific legal or regulatory requirement for annual influenza vaccination of healthcare personnel (including whether alternative measures like mask-wearing are required for unvaccinated staff); (2) The staff vaccination rate threshold that facilities achieving top-tier performance maintain; (3) The federal CMS requirement for designating an infection preventionist and the minimum time commitment required; (4) The specialized training requirement for the designated infection preventionist; (5) The federal OSHA mandate for offering hepatitis B vaccination to healthcare personnel; (6) State or federal requirements for ensuring healthcare personnel immunity or vaccination for measles, mumps, and rubella (MMR); (7) State or federal requirements for ensuring healthcare personnel immunity or vaccination for varicella (chickenpox); (8) Requirements or strong recommendations for Tdap (tetanus, diphtheria, pertussis) vaccination of healthcare personnel; (9) The federal requirement for reporting vaccination data to CDC's National Healthcare Safety Network (NHSN); (10) State health department licensing requirements for nursing homes; (11) Federal CMS certification requirements for Medicare/Medicaid participation; (12) Required policies for handling medical and religious exemptions from vaccination requirements; (13) Confirmation that the identified state was among those achieving 99% nursing home staff vaccination rates; (14) Reference citations for each regulatory requirement. Provide the name of the state and detailed documentation for all 14 requirement categories listed above.
"""

ALLOWED_STATES = ["Massachusetts", "Maine", "New York", "Rhode Island"]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RequirementItem(BaseModel):
    description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class InfluenzaPolicyInfo(BaseModel):
    policy_text: Optional[str] = None
    alt_measures: Optional[str] = None  # e.g., "Masks required for unvaccinated HCP during flu season"
    sources: List[str] = Field(default_factory=list)


class IPDesignationInfo(BaseModel):
    requirement_text: Optional[str] = None  # e.g., "Designate at least one infection preventionist"
    min_time_commitment: Optional[str] = None  # e.g., "at least part-time"
    sources: List[str] = Field(default_factory=list)


class StaffVaccinationPerformance(BaseModel):
    threshold_text: Optional[str] = None  # e.g., "Top-tier facilities maintain ~99% staff vaccination"
    sources: List[str] = Field(default_factory=list)


class ComplianceExtraction(BaseModel):
    # State identification and its supporting sources (2022 analyses)
    state_name: Optional[str] = None
    state_identification_sources: List[str] = Field(default_factory=list)

    # Category 1
    influenza_policy: Optional[InfluenzaPolicyInfo] = None
    # Category 2
    staff_vaccination_performance: Optional[StaffVaccinationPerformance] = None
    # Category 3
    ip_designation: Optional[IPDesignationInfo] = None
    # Category 4
    ip_training: Optional[RequirementItem] = None
    # Category 5
    hepatitis_b_osha: Optional[RequirementItem] = None
    # Category 6
    mmr_requirement: Optional[RequirementItem] = None
    # Category 7
    varicella_requirement: Optional[RequirementItem] = None
    # Category 8
    tdap_policy: Optional[RequirementItem] = None
    # Category 9
    nhsn_reporting: Optional[RequirementItem] = None
    # Category 10
    state_licensing: Optional[RequirementItem] = None
    # Category 11
    federal_cms_certification: Optional[RequirementItem] = None
    # Category 12
    exemption_policies: Optional[RequirementItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_compliance_doc() -> str:
    return """
    Extract the following structured information from the answer. Return null for any field not clearly present.
    Also extract all URLs explicitly cited in the answer for each item. Do not invent URLs. Include full protocol in URLs.

    1) state_name: The single U.S. state identified by the answer as one of the four states with 99% nursing home staff vaccination rates in 2022 (Massachusetts, Maine, New York, Rhode Island).
    2) state_identification_sources: All URLs cited to support that the identified state achieved ~99% nursing home staff vaccination in 2022.

    For each category below, extract both a concise description capturing the requirement exactly as stated in the answer and all supporting source URLs cited for that category.

    3) influenza_policy:
       - policy_text: The state's legal/regulatory requirement for annual influenza vaccination of healthcare personnel in nursing homes or equivalent settings (or the precise language used in the answer).
       - alt_measures: The answer's statement about any alternative measures for unvaccinated staff (e.g., mask-wearing), including whether they are required or permitted. If not addressed, return null.
       - sources: URLs cited for this policy.

    4) staff_vaccination_performance:
       - threshold_text: The staff vaccination rate level described for top-tier performance in the identified state (e.g., "99%" or equivalent).
       - sources: URLs cited for this performance statement.

    5) ip_designation:
       - requirement_text: The federal CMS requirement to designate an infection preventionist (IP).
       - min_time_commitment: The minimum time commitment required (e.g., "at least part-time") if the answer states it.
       - sources: URLs cited for this requirement.

    6) ip_training:
       - description: The specialized training requirement for the designated IP.
       - sources: URLs cited.

    7) hepatitis_b_osha:
       - description: The OSHA mandate to offer hepatitis B vaccination to employees with occupational exposure.
       - sources: URLs cited.

    8) mmr_requirement:
       - description: Requirement(s) for ensuring immunity or vaccination for measles, mumps, rubella (MMR) for healthcare personnel.
       - sources: URLs cited.

    9) varicella_requirement:
       - description: Requirement(s) for ensuring immunity or vaccination for varicella (chickenpox) for healthcare personnel.
       - sources: URLs cited.

    10) tdap_policy:
       - description: Requirement(s) or strong recommendations for Tdap vaccination of healthcare personnel.
       - sources: URLs cited.

    11) nhsn_reporting:
       - description: Federal requirement for nursing homes to report vaccination data to CDC's National Healthcare Safety Network (NHSN).
       - sources: URLs cited.

    12) state_licensing:
       - description: State health department licensing standards/requirements for nursing homes referenced by the answer.
       - sources: URLs cited.

    13) federal_cms_certification:
       - description: Federal CMS certification requirements for participation in Medicare/Medicaid referenced by the answer.
       - sources: URLs cited.

    14) exemption_policies:
       - description: Required policies for handling medical and religious vaccination exemptions (as stated in the answer).
       - sources: URLs cited.

    Return a single JSON object with the above fields using the exact field names specified.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _text_present(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len([u for u in urls if _text_present(u)]) > 0)


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls or []:
        if not _text_present(u):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_state_identification(evaluator: Evaluator, parent, data: ComplianceExtraction):
    node = evaluator.add_sequential(
        id="State_Identification",
        desc="Answer correctly identifies one of the four states (Massachusetts, Maine, New York, or Rhode Island) that achieved 99% nursing home staff vaccination rates in 2022",
        parent=parent,
        critical=True
    )

    exists = _text_present(data.state_name) and _has_sources(data.state_identification_sources)
    evaluator.add_custom_node(
        result=exists,
        id="state_id_provided",
        desc="State name is provided and supporting sources are cited",
        parent=node,
        critical=True
    )

    # Check state membership against allowed list (simple, no sources needed)
    state_check = evaluator.add_leaf(
        id="state_is_allowed",
        desc="Identified state is one of: Massachusetts, Maine, New York, or Rhode Island",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The identified state '{data.state_name or ''}' is one of the following: Massachusetts, Maine, New York, or Rhode Island.",
        node=state_check,
        additional_instruction="Allow case-insensitive comparison and minor variations (e.g., 'Mass.' is Massachusetts)."
    )

    # Verify 99% confirmation from 2022 published analyses/reports
    confirm_leaf = evaluator.add_leaf(
        id="state_99_confirmed_2022",
        desc="Sources confirm that the identified state achieved ~99% nursing home staff vaccination in 2022 (among the highest in the nation)",
        parent=node,
        critical=True
    )
    combined_sources = _dedup_urls(list(data.state_identification_sources or []) + (
        data.staff_vaccination_performance.sources if data.staff_vaccination_performance else []
    ))
    await evaluator.verify(
        claim=f"Published 2022 analyses or official reports show that {data.state_name or ''} achieved approximately 99% nursing home staff vaccination (i.e., among the highest in the nation).",
        node=confirm_leaf,
        sources=combined_sources,
        additional_instruction="Accept '99%' or 'approximately 99%' and phrasing that indicates the state is one of four states at ~99% for nursing home staff vaccination in 2022."
    )


async def verify_influenza_policy(evaluator: Evaluator, parent, data: ComplianceExtraction):
    node = evaluator.add_sequential(
        id="Influenza_Vaccination_Policy",
        desc="State legal/regulatory requirement for annual influenza vaccination of HCP (including alternative measures for unvaccinated)",
        parent=parent,
        critical=True
    )
    influenza = data.influenza_policy or InfluenzaPolicyInfo()

    provided = _text_present(influenza.policy_text) and _has_sources(influenza.sources) and _text_present(influenza.alt_measures)
    evaluator.add_custom_node(
        result=provided,
        id="influenza_info_provided",
        desc="Influenza policy and alternative-measures statement provided with sources",
        parent=node,
        critical=True
    )

    policy_leaf = evaluator.add_leaf(
        id="influenza_policy_supported",
        desc="Influenza vaccination policy is supported by cited sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In {data.state_name or 'the state'}, the policy/regulation for annual influenza vaccination of healthcare personnel in nursing homes (or equivalent settings) is: {influenza.policy_text or ''}.",
        node=policy_leaf,
        sources=_dedup_urls(influenza.sources),
        additional_instruction="Verify the cited sources substantiate the stated influenza vaccination policy for healthcare personnel in nursing homes (or broadly applicable HCP policies that apply to nursing homes). Wording can be equivalent."
    )

    alt_leaf = evaluator.add_leaf(
        id="influenza_alt_measures_supported",
        desc="Alternative measures for unvaccinated staff are correctly documented and supported",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The policy includes the following alternative compliance measure(s) for unvaccinated staff (e.g., mask-wearing): {influenza.alt_measures or ''}.",
        node=alt_leaf,
        sources=_dedup_urls(influenza.sources),
        additional_instruction="Check that the cited sources confirm whether alternative measures (e.g., mask use) are required, permitted, or specified for unvaccinated staff."
    )


async def verify_staff_vaccination_performance(evaluator: Evaluator, parent, data: ComplianceExtraction):
    node = evaluator.add_sequential(
        id="Staff_Vaccination_Performance",
        desc="Staff vaccination rate threshold/top-tier performance level for the identified state",
        parent=parent,
        critical=True
    )
    perf = data.staff_vaccination_performance or StaffVaccinationPerformance()

    provided = _text_present(perf.threshold_text) and _has_sources(perf.sources)
    evaluator.add_custom_node(
        result=provided,
        id="staff_perf_provided",
        desc="Staff vaccination performance/threshold provided with sources",
        parent=node,
        critical=True
    )

    perf_leaf = evaluator.add_leaf(
        id="staff_perf_supported",
        desc="Staff vaccination performance threshold is supported by sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In {data.state_name or 'the state'}, facilities achieving top-tier performance maintain the following staff vaccination rate threshold or level: {perf.threshold_text or ''}.",
        node=perf_leaf,
        sources=_dedup_urls(perf.sources),
        additional_instruction="Verify that the cited sources support the claimed threshold/level (e.g., ~99%). Equivalent phrasing is acceptable."
    )


async def verify_ip_designation(evaluator: Evaluator, parent, data: ComplianceExtraction):
    node = evaluator.add_sequential(
        id="Infection_Preventionist_Designation",
        desc="Federal CMS requirement to designate at least one infection preventionist and the minimum time commitment",
        parent=parent,
        critical=True
    )
    ipd = data.ip_designation or IPDesignationInfo()

    provided = _text_present(ipd.requirement_text) and _text_present(ipd.min_time_commitment) and _has_sources(ipd.sources)
    evaluator.add_custom_node(
        result=provided,
        id="ip_designation_provided",
        desc="IP designation requirement and time commitment provided with sources",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="ip_designation_supported",
        desc="IP designation and time commitment are supported by sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Under federal CMS requirements, nursing homes must designate at least one infection preventionist with a minimum time commitment of {ipd.min_time_commitment or ''}. Stated requirement: {ipd.requirement_text or ''}.",
        node=leaf,
        sources=_dedup_urls(ipd.sources),
        additional_instruction="Verify against federal long-term care infection control regulations (e.g., 42 CFR 483.80) or official CMS guidance. Equivalent wording is acceptable."
    )


async def verify_ip_training(evaluator: Evaluator, parent, data: ComplianceExtraction):
    node = evaluator.add_sequential(
        id="IP_Training_Requirement",
        desc="Specialized training requirement for the designated infection preventionist",
        parent=parent,
        critical=True
    )
    item = data.ip_training or RequirementItem()

    provided = _text_present(item.description) and _has_sources(item.sources)
    evaluator.add_custom_node(
        result=provided,
        id="ip_training_provided",
        desc="IP specialized training requirement provided with sources",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="ip_training_supported",
        desc="IP specialized training requirement supported by sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The designated infection preventionist must complete specialized training in infection prevention and control: {item.description or ''}.",
        node=leaf,
        sources=_dedup_urls(item.sources),
        additional_instruction="Verify that CMS or official training requirements/guidance supports the training requirement stated."
    )


async def verify_simple_requirement(evaluator: Evaluator, parent, cat_id: str, cat_desc: str, item: RequirementItem, add_ins: str = "Verify that the cited source(s) explicitly support this requirement; equivalent wording acceptable."):
    node = evaluator.add_sequential(
        id=cat_id,
        desc=cat_desc,
        parent=parent,
        critical=True
    )

    provided = _text_present(item.description) and _has_sources(item.sources)
    evaluator.add_custom_node(
        result=provided,
        id=f"{cat_id}_provided",
        desc=f"{cat_desc} is provided with sources",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"{cat_id}_supported",
        desc=f"{cat_desc} is supported by cited sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=item.description or "",
        node=leaf,
        sources=_dedup_urls(item.sources),
        additional_instruction=add_ins
    )


async def verify_reference_citations(evaluator: Evaluator, parent, data: ComplianceExtraction):
    # Ensure every category includes at least one citation URL
    node = evaluator.add_parallel(
        id="Reference_Citations",
        desc="All requirement categories include at least one supporting citation from official or authoritative sources",
        parent=parent,
        critical=True
    )

    def add_has_sources_check(node_id: str, label: str, urls: List[str]):
        evaluator.add_custom_node(
            result=_has_sources(urls),
            id=node_id,
            desc=f"{label} has at least one citation URL",
            parent=node,
            critical=True
        )

    add_has_sources_check("refs_state_identification", "State identification (99% confirmation)", data.state_identification_sources)
    add_has_sources_check("refs_influenza_policy", "Influenza vaccination policy", (data.influenza_policy.sources if data.influenza_policy else []))
    add_has_sources_check("refs_staff_performance", "Staff vaccination performance", (data.staff_vaccination_performance.sources if data.staff_vaccination_performance else []))
    add_has_sources_check("refs_ip_designation", "Infection preventionist designation", (data.ip_designation.sources if data.ip_designation else []))
    add_has_sources_check("refs_ip_training", "Infection preventionist training", (data.ip_training.sources if data.ip_training else []))
    add_has_sources_check("refs_hepb_osha", "OSHA Hepatitis B vaccination mandate", (data.hepatitis_b_osha.sources if data.hepatitis_b_osha else []))
    add_has_sources_check("refs_mmr", "MMR requirement", (data.mmr_requirement.sources if data.mmr_requirement else []))
    add_has_sources_check("refs_varicella", "Varicella requirement", (data.varicella_requirement.sources if data.varicella_requirement else []))
    add_has_sources_check("refs_tdap", "Tdap policy", (data.tdap_policy.sources if data.tdap_policy else []))
    add_has_sources_check("refs_nhsn", "NHSN reporting requirement", (data.nhsn_reporting.sources if data.nhsn_reporting else []))
    add_has_sources_check("refs_state_licensing", "State licensing standards", (data.state_licensing.sources if data.state_licensing else []))
    add_has_sources_check("refs_cms_cert", "Federal CMS certification", (data.federal_cms_certification.sources if data.federal_cms_certification else []))
    add_has_sources_check("refs_exemptions", "Exemption policy requirements", (data.exemption_policies.sources if data.exemption_policies else []))


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
    Evaluate an answer against the nursing home vaccination and regulatory requirements rubric.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    _ = evaluator.initialize(
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

    # Create a critical main node under the framework root (to honor rubric root criticality)
    main = evaluator.add_parallel(
        id="Root",
        desc="Answer provides comprehensive documentation of all vaccination and regulatory compliance requirements for nursing homes in one of the four states that achieved 99% staff vaccination rates",
        parent=evaluator.root,
        critical=True
    )

    # Add ground-truth context (allowed states list)
    evaluator.add_ground_truth({
        "allowed_states_2022_99_percent": ALLOWED_STATES,
        "notes": "Any identified state must be one of these four; all categories must be documented with sources."
    }, gt_type="ground_truth")

    # Extract structured information from the answer
    data: ComplianceExtraction = await evaluator.extract(
        prompt=prompt_extract_compliance_doc(),
        template_class=ComplianceExtraction,
        extraction_name="compliance_extraction"
    )

    # Build verification tree according to rubric
    await verify_state_identification(evaluator, main, data)
    await verify_influenza_policy(evaluator, main, data)
    await verify_staff_vaccination_performance(evaluator, main, data)
    await verify_ip_designation(evaluator, main, data)
    await verify_ip_training(evaluator, main, data)

    # The following categories use a common simple verification pattern
    await verify_simple_requirement(
        evaluator, main,
        cat_id="Hepatitis_B_OSHA_Mandate",
        cat_desc="Federal OSHA requirement to offer hepatitis B vaccination to employees with occupational exposure",
        item=data.hepatitis_b_osha or RequirementItem(),
        add_ins="Verify against OSHA Bloodborne Pathogens Standard (29 CFR 1910.1030) or official OSHA publications that the employer must offer Hepatitis B vaccination at no cost to employees with occupational exposure."
    )

    await verify_simple_requirement(
        evaluator, main,
        cat_id="MMR_Vaccination_Requirement",
        cat_desc="Requirement for ensuring healthcare personnel immunity or vaccination for MMR",
        item=data.mmr_requirement or RequirementItem(),
        add_ins="Verify that the cited source(s) require or clearly direct facilities to ensure immunity or vaccination for measles, mumps, and rubella for healthcare personnel. State or federal authoritative sources are acceptable."
    )

    await verify_simple_requirement(
        evaluator, main,
        cat_id="Varicella_Vaccination_Requirement",
        cat_desc="Requirement for ensuring healthcare personnel immunity or vaccination for varicella",
        item=data.varicella_requirement or RequirementItem(),
        add_ins="Verify that the cited source(s) require or clearly direct facilities to ensure immunity or vaccination for varicella (chickenpox) for healthcare personnel."
    )

    await verify_simple_requirement(
        evaluator, main,
        cat_id="Tdap_Vaccination_Policy",
        cat_desc="Requirements or strong recommendations for Tdap vaccination of healthcare personnel",
        item=data.tdap_policy or RequirementItem(),
        add_ins="Verify that the cited source(s) describe requirements or strong recommendations for Tdap vaccination of healthcare personnel (state regs or authoritative CDC/ACIP guidance accepted if applicable to nursing homes)."
    )

    await verify_simple_requirement(
        evaluator, main,
        cat_id="NHSN_Reporting_Requirement",
        cat_desc="Federal requirement for nursing homes to report vaccination data to CDC's NHSN",
        item=data.nhsn_reporting or RequirementItem(),
        add_ins="Verify that the cited federal source(s) (e.g., CMS or CDC/NHSN) require nursing homes to report vaccination data to NHSN."
    )

    await verify_simple_requirement(
        evaluator, main,
        cat_id="State_Licensing_Standards",
        cat_desc="State health department licensing requirements for nursing homes",
        item=data.state_licensing or RequirementItem(),
        add_ins=f"Verify that the cited state source(s) specify licensing standards/requirements for nursing homes in {data.state_name or 'the state'}."
    )

    await verify_simple_requirement(
        evaluator, main,
        cat_id="Federal_CMS_Certification",
        cat_desc="Federal CMS certification requirements for Medicare/Medicaid participation",
        item=data.federal_cms_certification or RequirementItem(),
        add_ins="Verify against CMS regulations/guidance that set certification requirements for nursing homes participating in Medicare/Medicaid (e.g., 42 CFR Part 483)."
    )

    await verify_simple_requirement(
        evaluator, main,
        cat_id="Exemption_Policy_Requirements",
        cat_desc="Policies for handling medical and religious vaccination exemptions",
        item=data.exemption_policies or RequirementItem(),
        add_ins="Verify that the cited source(s) indicate facilities must maintain policies addressing medical and religious exemptions from vaccination requirements (or otherwise handling exemptions consistent with law)."
    )

    # Reference citations completeness check
    await verify_reference_citations(evaluator, main, data)

    # Return standardized evaluation summary
    return evaluator.get_summary()