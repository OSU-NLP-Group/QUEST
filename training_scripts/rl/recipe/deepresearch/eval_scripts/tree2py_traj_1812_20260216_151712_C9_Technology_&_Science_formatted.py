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
TASK_ID = "telecom_ca_backup_power"
TASK_DESCRIPTION = """For facilities-based wireless telecommunications carriers operating macro cell towers in California's Tier 3 High Fire Threat Districts, provide a comprehensive compliance analysis that addresses the following:

1. What is the legally required minimum backup power duration for these cell towers, and which specific California Public Utilities Commission (CPUC) decision established this requirement? Include the decision number, effective date, and the implementation deadline that carriers had to meet.

2. What are the three categories of permitted exceptions under which a carrier may be exempt from or modify the backup power requirement? Briefly explain each exception category.

3. What regulatory documentation must wireless carriers file with the CPUC to maintain compliance with backup power requirements in High Fire Threat Districts? List all required plans, reports, and notifications.

4. How do current federal (FCC) backup power requirements differ from California's requirements? Specifically address: (a) the current status of the FCC's 2007 backup power rules that originally required 8 hours for cell sites, and (b) what limited federal backup power requirements, if any, currently apply to wireless carriers.

5. If a wireless carrier experiences a reportable outage affecting backup power or 911 service at a Tier 3 HFTD cell tower, what are the FCC's Network Outage Reporting System (NORS) reporting timelines for both initial and final reports?

For each answer component, cite the specific regulatory source (decision number, CFR section, or official document) that supports your response.
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class Part1Extraction(BaseModel):
    required_duration: Optional[str] = None
    cpu_c_decision_number: Optional[str] = None
    effective_date: Optional[str] = None
    implementation_deadline: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Part2Extraction(BaseModel):
    exception_category_1: Optional[str] = None
    exception_category_2: Optional[str] = None
    exception_category_3: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Part3Extraction(BaseModel):
    resiliency_plan_statement: Optional[str] = None
    emergency_operations_plan_statement: Optional[str] = None
    annual_compliance_report_statement: Optional[str] = None
    notifications_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Part4Extraction(BaseModel):
    fcc_2007_rules_status: Optional[str] = None
    fcc_current_scope_limitation: Optional[str] = None
    fcc_24_hour_requirement_psap_admin_lines: Optional[str] = None
    fcc_72_hour_requirement_selective_routers: Optional[str] = None
    fcc_annual_certification_filing: Optional[str] = None
    california_hftd_requirement: Optional[str] = None
    federal_vs_california_difference: Optional[str] = None
    federal_sources: List[str] = Field(default_factory=list)
    california_sources: List[str] = Field(default_factory=list)


class Part5Extraction(BaseModel):
    nors_initial_report_timeline: Optional[str] = None
    nors_final_report_timeline: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ComplianceExtraction(BaseModel):
    part1: Part1Extraction = Field(default_factory=Part1Extraction)
    part2: Part2Extraction = Field(default_factory=Part2Extraction)
    part3: Part3Extraction = Field(default_factory=Part3Extraction)
    part4: Part4Extraction = Field(default_factory=Part4Extraction)
    part5: Part5Extraction = Field(default_factory=Part5Extraction)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_compliance() -> str:
    return """
Extract the following structured information exactly as it appears in the answer. Do not invent any content. If a field is missing, set it to null. For any sources fields, only extract URLs explicitly present in the answer (plain or markdown). Do not infer URLs from decision numbers or citations.

Return a JSON object with keys: part1, part2, part3, part4, part5. Each part must follow the field requirements below.

part1:
- required_duration: the stated minimum backup power duration for covered wireless macro sites in Tier 3 HFTDs (e.g., "72 hours").
- cpu_c_decision_number: the decision number establishing the requirement (e.g., "D.20-07-011").
- effective_date: the stated effective date of that decision (e.g., "July 16, 2020").
- implementation_deadline: the stated implementation deadline (e.g., "12 months from effective date" or "by July 2021").
- sources: array of URLs cited that support the above (extract only URLs explicitly present).

part2:
- exception_category_1: the explained "no backup needed" exception category text from the answer.
- exception_category_2: the explained "safety risks" exception category text from the answer.
- exception_category_3: the explained "objectively impossible or infeasible" exception category text from the answer.
- sources: array of URLs cited supporting the exception categories.

part3:
- resiliency_plan_statement: the answer’s statement about filing resiliency plans with CPUC.
- emergency_operations_plan_statement: the answer’s statement about filing emergency operations plans with CPUC.
- annual_compliance_report_statement: the answer’s statement about filing annual compliance reports.
- notifications_statement: the answer’s statement about CPUC-required notifications relevant to backup power compliance (if the answer states none are identified, extract that text).
- sources: array of URLs cited supporting the filings/requirements.

part4:
- fcc_2007_rules_status: the stated current status of the FCC’s 2007 backup power rules that originally required 8 hours for cell sites.
- fcc_current_scope_limitation: the stated scope of current FCC backup power rules (e.g., limited to covered 911 service providers).
- fcc_24_hour_requirement_psap_admin_lines: the stated 24-hour backup power requirement for covered 911 service providers’ central offices serving PSAP admin lines.
- fcc_72_hour_requirement_selective_routers: the stated 72-hour backup power requirement for covered 911 service providers’ central offices with selective routers for 911.
- fcc_annual_certification_filing: the stated requirement for covered 911 service providers to file annual certifications with the FCC.
- california_hftd_requirement: the stated California requirement (e.g., 72 hours for Tier 2 and Tier 3 HFTDs).
- federal_vs_california_difference: the explicit difference explanation provided in the answer.
- federal_sources: array of URLs cited that support the federal statements (CFR sections, FCC orders).
- california_sources: array of URLs cited that support the California statements (CPUC decisions or official CPUC pages).

part5:
- nors_initial_report_timeline: the stated NORS initial report timeline (e.g., "within 72 hours of discovery").
- nors_final_report_timeline: the stated NORS final report timeline (e.g., "within 30 days of discovery").
- sources: array of URLs cited (e.g., 47 CFR Part 4).

Notes:
- Extract strings verbatim as written in the answer where feasible. Normalize whitespace but keep the original meaning.
- For all sources arrays, include only valid URLs explicitly mentioned in the answer; if none are present, return an empty array.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(val: Optional[str], placeholder: str = "UNSPECIFIED") -> str:
    return val.strip() if isinstance(val, str) and val.strip() else placeholder


def _evidence_instruction(urls: List[str]) -> str:
    if urls and len(urls) > 0:
        return ("Use only the provided URLs as evidence. Mark as Correct only if at least one URL explicitly "
                "supports the claim as stated. Prefer explicit statements over inference.")
    else:
        return ("IMPORTANT: The answer did not provide any URL sources for this claim. In this evaluation framework, "
                "factual claims must be supported by cited URLs. Therefore, you must mark the claim as Incorrect / "
                "Not Supported due to missing evidence, regardless of the answer text.")


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_part1(evaluator: Evaluator, parent_node, data: Part1Extraction) -> None:
    part_node = evaluator.add_parallel(
        id="Part_1_Duration_CPUC_Decision_and_Deadline",
        desc="Answer Part 1: required backup duration and the CPUC decision establishing it (decision number, effective date, and implementation deadline)",
        parent=parent_node,
        critical=True
    )
    urls = data.sources
    add_ins = _evidence_instruction(urls)

    # Required_Duration
    node_duration = evaluator.add_leaf(
        id="Required_Duration",
        desc="State the legally required minimum backup power duration for the covered wireless facilities in Tier 3 HFTDs",
        parent=part_node,
        critical=True
    )
    claim_duration = f"The legally required minimum backup power duration for covered wireless macro cell facilities in California Tier 3 HFTDs is {_safe(data.required_duration)}."
    await evaluator.verify(claim=claim_duration, node=node_duration, sources=urls, additional_instruction=add_ins)

    # CPUC_Decision_Number
    node_decision = evaluator.add_leaf(
        id="CPUC_Decision_Number",
        desc="Identify the CPUC decision establishing the requirement",
        parent=part_node,
        critical=True
    )
    claim_decision = f"The CPUC decision establishing this backup power requirement is {_safe(data.cpu_c_decision_number)}."
    await evaluator.verify(claim=claim_decision, node=node_decision, sources=urls, additional_instruction=add_ins)

    # Effective_Date
    node_effective = evaluator.add_leaf(
        id="Effective_Date",
        desc="Provide the effective date of the decision",
        parent=part_node,
        critical=True
    )
    claim_effective = f"The effective date of CPUC decision {_safe(data.cpu_c_decision_number)} is {_safe(data.effective_date)}."
    await evaluator.verify(claim=claim_effective, node=node_effective, sources=urls, additional_instruction=add_ins)

    # Implementation_Deadline
    node_deadline = evaluator.add_leaf(
        id="Implementation_Deadline",
        desc="Provide the implementation deadline",
        parent=part_node,
        critical=True
    )
    claim_deadline = f"The implementation deadline for carriers to meet this requirement was {_safe(data.implementation_deadline)}."
    await evaluator.verify(claim=claim_deadline, node=node_deadline, sources=urls, additional_instruction=add_ins)

    # Part_1_Source_Citations
    node_sources = evaluator.add_leaf(
        id="Part_1_Source_Citations",
        desc="Cite the specific CPUC decision/source that supports the duration, effective date, and deadline statements",
        parent=part_node,
        critical=True
    )
    claim_sources = (
        f"CPUC decision {_safe(data.cpu_c_decision_number)} established a {_safe(data.required_duration)} backup power requirement, "
        f"effective {_safe(data.effective_date)}, with implementation deadline {_safe(data.implementation_deadline)}."
    )
    await evaluator.verify(claim=claim_sources, node=node_sources, sources=urls, additional_instruction=add_ins)


async def verify_part2(evaluator: Evaluator, parent_node, data: Part2Extraction) -> None:
    part_node = evaluator.add_parallel(
        id="Part_2_Three_Exception_Categories",
        desc="Answer Part 2: identify and briefly explain the three permitted exception categories",
        parent=parent_node,
        critical=True
    )
    urls = data.sources
    add_ins = _evidence_instruction(urls)

    # Exception_Category_1_No_Backup_Needed
    node_exc1 = evaluator.add_leaf(
        id="Exception_Category_1_No_Backup_Needed",
        desc="Explain the exception category where the facility does not need backup power",
        parent=part_node,
        critical=True
        )
    claim_exc1 = (
        f"The CPUC decision includes an exception category in which a facility does not need backup power. "
        f"The answer explains it as: {_safe(data.exception_category_1)}."
    )
    await evaluator.verify(claim=claim_exc1, node=node_exc1, sources=urls, additional_instruction=add_ins)

    # Exception_Category_2_Safety_Risks
    node_exc2 = evaluator.add_leaf(
        id="Exception_Category_2_Safety_Risks",
        desc="Explain the exception category where the facility cannot support backup power due to safety risks",
        parent=part_node,
        critical=True
    )
    claim_exc2 = (
        f"The CPUC decision includes an exception category where backup power cannot be supported due to safety risks. "
        f"The answer explains it as: {_safe(data.exception_category_2)}."
    )
    await evaluator.verify(claim=claim_exc2, node=node_exc2, sources=urls, additional_instruction=add_ins)

    # Exception_Category_3_Objectively_Impossible_or_Infeasible
    node_exc3 = evaluator.add_leaf(
        id="Exception_Category_3_Objectively_Impossible_or_Infeasible",
        desc="Explain the exception category where deployment is objectively impossible or infeasible",
        parent=part_node,
        critical=True
    )
    claim_exc3 = (
        f"The CPUC decision includes an exception category where deployment is objectively impossible or infeasible. "
        f"The answer explains it as: {_safe(data.exception_category_3)}."
    )
    await evaluator.verify(claim=claim_exc3, node=node_exc3, sources=urls, additional_instruction=add_ins)

    # Part_2_Source_Citations
    node_sources = evaluator.add_leaf(
        id="Part_2_Source_Citations",
        desc="Cite the CPUC decision/source that establishes these exception categories",
        parent=part_node,
        critical=True
    )
    claim_sources = "The cited CPUC decision establishes three exception categories: no backup needed, safety risks, and objectively impossible or infeasible."
    await evaluator.verify(claim=claim_sources, node=node_sources, sources=urls, additional_instruction=add_ins)


async def verify_part3(evaluator: Evaluator, parent_node, data: Part3Extraction) -> None:
    part_node = evaluator.add_parallel(
        id="Part_3_CPUC_Required_Documentation",
        desc="Answer Part 3: identify CPUC filings required to maintain compliance (plans/reports/notifications) and cite sources",
        parent=parent_node,
        critical=True
    )
    urls = data.sources
    add_ins = _evidence_instruction(urls)

    # CPUC_Resiliency_Plan
    node_rp = evaluator.add_leaf(
        id="CPUC_Resiliency_Plan",
        desc="State that carriers must file resiliency plans with the CPUC detailing backup power strategies",
        parent=part_node,
        critical=True
    )
    claim_rp = f"The CPUC requires wireless carriers to file resiliency plans detailing backup power strategies. Statement: {_safe(data.resiliency_plan_statement)}."
    await evaluator.verify(claim=claim_rp, node=node_rp, sources=urls, additional_instruction=add_ins)

    # CPUC_Emergency_Operations_Plan
    node_eop = evaluator.add_leaf(
        id="CPUC_Emergency_Operations_Plan",
        desc="State that carriers must file emergency operations plans with the CPUC",
        parent=part_node,
        critical=True
    )
    claim_eop = f"The CPUC requires wireless carriers to file emergency operations plans. Statement: {_safe(data.emergency_operations_plan_statement)}."
    await evaluator.verify(claim=claim_eop, node=node_eop, sources=urls, additional_instruction=add_ins)

    # CPUC_Annual_Compliance_Report
    node_annual = evaluator.add_leaf(
        id="CPUC_Annual_Compliance_Report",
        desc="State that carriers must file annual compliance reports with the CPUC regarding backup power implementation",
        parent=part_node,
        critical=True
    )
    claim_annual = f"The CPUC requires annual compliance reports regarding backup power implementation. Statement: {_safe(data.annual_compliance_report_statement)}."
    await evaluator.verify(claim=claim_annual, node=node_annual, sources=urls, additional_instruction=add_ins)

    # CPUC_Required_Notifications
    node_notif = evaluator.add_leaf(
        id="CPUC_Required_Notifications",
        desc="Address CPUC-required notifications relevant to backup power compliance",
        parent=part_node,
        critical=True
    )
    claim_notif = f"The answer addresses CPUC-required notifications relevant to backup power compliance as: {_safe(data.notifications_statement)}."
    await evaluator.verify(claim=claim_notif, node=node_notif, sources=urls, additional_instruction=add_ins)

    # Part_3_Source_Citations
    node_sources = evaluator.add_leaf(
        id="Part_3_Source_Citations",
        desc="Cite the CPUC decision(s)/official CPUC requirements that support each stated filing obligation",
        parent=part_node,
        critical=True
    )
    claim_sources = "The cited CPUC sources require carriers to file resiliency plans, emergency operations plans, annual compliance reports, and any stated notifications relevant to backup power compliance."
    await evaluator.verify(claim=claim_sources, node=node_sources, sources=urls, additional_instruction=add_ins)


async def verify_part4(evaluator: Evaluator, parent_node, data: Part4Extraction) -> None:
    part_node = evaluator.add_parallel(
        id="Part_4_Federal_vs_California_Requirements",
        desc="Answer Part 4: compare current federal (FCC) backup power requirements to California CPUC requirements, including status of the FCC 2007 rules",
        parent=parent_node,
        critical=True
    )

    fed_urls = data.federal_sources
    ca_urls = data.california_sources
    add_ins_fed = _evidence_instruction(fed_urls)
    add_ins_ca = _evidence_instruction(ca_urls)
    add_ins_both = _evidence_instruction((fed_urls or []) + (ca_urls or []))

    # FCC_2007_Rules_Status
    node_fcc2007 = evaluator.add_leaf(
        id="FCC_2007_Rules_Status",
        desc="State that the FCC's 2007 backup power rules were vacated and deleted in 2011 and are not currently in effect as general requirements",
        parent=part_node,
        critical=True
    )
    claim_fcc2007 = f"The status of the FCC's 2007 backup power rules is: {_safe(data.fcc_2007_rules_status)}."
    await evaluator.verify(claim=claim_fcc2007, node=node_fcc2007, sources=fed_urls, additional_instruction=add_ins_fed)

    # FCC_Current_Scope_Limitation
    node_scope = evaluator.add_leaf(
        id="FCC_Current_Scope_Limitation",
        desc="State that current FCC backup power requirements apply only to covered 911 service providers",
        parent=part_node,
        critical=True
    )
    claim_scope = f"The current FCC backup power requirements apply to covered 911 service providers (not all carriers). Stated scope: {_safe(data.fcc_current_scope_limitation)}."
    await evaluator.verify(claim=claim_scope, node=node_scope, sources=fed_urls, additional_instruction=add_ins_fed)

    # FCC_24_Hour_Requirement_PSAP_Admin_Lines
    node_24 = evaluator.add_leaf(
        id="FCC_24_Hour_Requirement_PSAP_Admin_Lines",
        desc="State the 24-hour backup power requirement for covered 911 service providers for central offices providing PSAP admin lines",
        parent=part_node,
        critical=True
    )
    claim_24 = f"The FCC requires 24 hours of backup power for covered 911 providers' central offices serving PSAP administrative lines. Statement: {_safe(data.fcc_24_hour_requirement_psap_admin_lines)}."
    await evaluator.verify(claim=claim_24, node=node_24, sources=fed_urls, additional_instruction=add_ins_fed)

    # FCC_72_Hour_Requirement_Selective_Routers
    node_72 = evaluator.add_leaf(
        id="FCC_72_Hour_Requirement_Selective_Routers",
        desc="State the 72-hour backup power requirement for covered 911 service providers for central offices that contain selective routers",
        parent=part_node,
        critical=True
    )
    claim_72 = f"The FCC requires 72 hours of backup power for covered 911 providers' central offices containing selective routers for 911. Statement: {_safe(data.fcc_72_hour_requirement_selective_routers)}."
    await evaluator.verify(claim=claim_72, node=node_72, sources=fed_urls, additional_instruction=add_ins_fed)

    # FCC_Annual_Certification_Filing
    node_cert = evaluator.add_leaf(
        id="FCC_Annual_Certification_Filing",
        desc="State that covered 911 service providers must file annual certifications with the FCC regarding compliance",
        parent=part_node,
        critical=True
    )
    claim_cert = f"Covered 911 service providers must file annual certifications with the FCC regarding backup power standards. Statement: {_safe(data.fcc_annual_certification_filing)}."
    await evaluator.verify(claim=claim_cert, node=node_cert, sources=fed_urls, additional_instruction=add_ins_fed)

    # California_Wireless_HFTD_Requirement
    node_ca = evaluator.add_leaf(
        id="California_Wireless_HFTD_Requirement",
        desc="State that California CPUC requires 72-hour backup power for facilities-based wireless providers in Tier 2 and Tier 3 HFTDs",
        parent=part_node,
        critical=True
    )
    claim_ca = f"In California, CPUC requires 72-hour backup power for facilities-based wireless providers in Tier 2 and Tier 3 HFTDs. Statement: {_safe(data.california_hftd_requirement)}."
    await evaluator.verify(claim=claim_ca, node=node_ca, sources=ca_urls, additional_instruction=add_ins_ca)

    # Federal_vs_California_Differences_Explained
    node_diff = evaluator.add_leaf(
        id="Federal_vs_California_Differences_Explained",
        desc="Explicitly explain at least one concrete difference between current federal requirements and California’s requirements",
        parent=part_node,
        critical=True
    )
    claim_diff = (
        f"The explanation of differences between federal and California backup power requirements is: "
        f"{_safe(data.federal_vs_california_difference)}. This explanation is consistent with the cited federal and California sources."
    )
    await evaluator.verify(claim=claim_diff, node=node_diff, sources=(fed_urls + ca_urls), additional_instruction=add_ins_both)

    # Part_4_Source_Citations
    node_sources = evaluator.add_leaf(
        id="Part_4_Source_Citations",
        desc="Cite relevant CFR/FCC orders for federal statements and the relevant CPUC decision for California statements",
        parent=part_node,
        critical=True
    )
    claim_sources = "The cited federal sources (CFR sections/FCC orders) and California CPUC decision(s) support the statements made about backup power requirements."
    await evaluator.verify(claim=claim_sources, node=node_sources, sources=(fed_urls + ca_urls), additional_instruction=add_ins_both)


async def verify_part5(evaluator: Evaluator, parent_node, data: Part5Extraction) -> None:
    part_node = evaluator.add_parallel(
        id="Part_5_FCC_NORS_Reporting_Timelines",
        desc="Answer Part 5: provide FCC NORS initial and final reporting timelines for a reportable outage",
        parent=parent_node,
        critical=True
    )
    urls = data.sources
    add_ins = _evidence_instruction(urls)

    # NORS_Initial_Report_Timeline
    node_initial = evaluator.add_leaf(
        id="NORS_Initial_Report_Timeline",
        desc="State the initial NORS report timeline",
        parent=part_node,
        critical=True
    )
    claim_initial = f"The initial NORS report timeline is {_safe(data.nors_initial_report_timeline)}."
    await evaluator.verify(claim=claim_initial, node=node_initial, sources=urls, additional_instruction=add_ins)

    # NORS_Final_Report_Timeline
    node_final = evaluator.add_leaf(
        id="NORS_Final_Report_Timeline",
        desc="State the final NORS report timeline",
        parent=part_node,
        critical=True
    )
    claim_final = f"The final NORS report timeline is {_safe(data.nors_final_report_timeline)}."
    await evaluator.verify(claim=claim_final, node=node_final, sources=urls, additional_instruction=add_ins)

    # Part_5_Source_Citations
    node_sources = evaluator.add_leaf(
        id="Part_5_Source_Citations",
        desc="Cite the relevant 47 CFR Part 4 provision(s) supporting the initial and final NORS timelines",
        parent=part_node,
        critical=True
    )
    claim_sources = "The relevant 47 CFR Part 4 provisions support the stated initial and final NORS reporting timelines."
    await evaluator.verify(claim=claim_sources, node=node_sources, sources=urls, additional_instruction=add_ins)


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
    Evaluate an answer for the California HFTD wireless backup power compliance analysis.
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
        default_model=model
    )

    # Extraction
    extracted: ComplianceExtraction = await evaluator.extract(
        prompt=prompt_extract_compliance(),
        template_class=ComplianceExtraction,
        extraction_name="compliance_extraction"
    )

    # Optional ground truth/context (not used for hard matching; for report context only)
    evaluator.add_ground_truth({
        "expected_reference_points": {
            "cpuc_decision_example": "D.20-07-011 (effective July 16, 2020; implementation within 12 months)",
            "ca_requirement_example": "72-hour backup power for Tier 2/3 HFTDs for facilities-based wireless",
            "fcc_2007_rules_status_example": "Vacated and subsequently deleted as general rules in 2011",
            "fcc_current_scope_example": "Covered 911 service providers with 24/72-hour CO requirements; annual certifications",
            "nors_timelines_example": "Initial within 72 hours; final within 30 days (47 CFR Part 4)"
        }
    })

    # Build top-level critical node
    top_node = evaluator.add_parallel(
        id="Telecommunications_Backup_Power_Compliance_Analysis",
        desc="Comprehensive analysis of backup power requirements for facilities-based wireless carriers operating macro cell towers in California Tier 3 High Fire Threat Districts",
        parent=root,
        critical=True
    )

    # Verify each part according to rubric
    await verify_part1(evaluator, top_node, extracted.part1)
    await verify_part2(evaluator, top_node, extracted.part2)
    await verify_part3(evaluator, top_node, extracted.part3)
    await verify_part4(evaluator, top_node, extracted.part4)
    await verify_part5(evaluator, top_node, extracted.part5)

    return evaluator.get_summary()