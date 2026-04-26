import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pmp_cissp_cert_requirements"
TASK_DESCRIPTION = (
    "You are planning to advance your career in IT management and want to obtain two major professional certifications: "
    "the Project Management Professional (PMP) certification and the Certified Information Systems Security Professional (CISSP) certification. "
    "You currently hold a bachelor's degree in Information Technology and have 5 years of work experience in IT project management and security.\n\n"
    "For each certification, provide the following information:\n\n"
    "For PMP Certification:\n"
    "1. The required months of project management experience for candidates with a four-year degree, and the timeframe within which this experience must have been gained\n"
    "2. The number of contact hours of project management education required before taking the exam\n"
    "3. The total number of Professional Development Units (PDUs) required for renewal and the length of the renewal cycle\n"
    "4. The minimum number of PDUs that must come from the Education category\n"
    "5. The minimum number of PDUs required in each skill area of the PMI Talent Triangle, and list the three skill area categories\n"
    "6. The exam cost for both PMI members and non-members\n\n"
    "For CISSP Certification:\n"
    "1. The minimum number of years of cumulative, full-time work experience required, the minimum number of CISSP domains in which experience is required, and the total number of CISSP domains\n"
    "2. The number of years of experience that can be waived with a four-year degree\n"
    "3. The total number of Continuing Professional Education (CPE) credits required for renewal and the length of the renewal cycle\n"
    "4. The minimum number of CPE credits that must be Group A (domain-related) and explain what Group A credits represent\n"
    "5. The recommended number of CPEs to earn annually\n"
    "6. The exam cost\n\n"
    "For all requirements, provide reference URLs from official certification bodies or authoritative sources that document each requirement."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PMPExperience(BaseModel):
    months_required: Optional[str] = None
    timeframe_window: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class PMPEducation(BaseModel):
    contact_hours_required: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class PMPRenewal(BaseModel):
    total_pdus_required: Optional[str] = None
    cycle_length: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class PMPEducationMinimum(BaseModel):
    min_education_pdus: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class PMPTalentTriangle(BaseModel):
    min_pdus_per_area: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class PMPExamCost(BaseModel):
    member_cost: Optional[str] = None
    non_member_cost: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class PMPInfo(BaseModel):
    experience: Optional[PMPExperience] = None
    education: Optional[PMPEducation] = None
    renewal: Optional[PMPRenewal] = None
    education_minimum: Optional[PMPEducationMinimum] = None
    talent_triangle: Optional[PMPTalentTriangle] = None
    exam_cost: Optional[PMPExamCost] = None


class CISSPExperienceRequirement(BaseModel):
    min_years: Optional[str] = None
    min_domains_experience: Optional[str] = None
    total_domains: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class CISSPEducationWaiver(BaseModel):
    waiver_years_with_degree: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class CISSPRenewal(BaseModel):
    total_cpes_required: Optional[str] = None
    cycle_length: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class CISSPGroupA(BaseModel):
    min_group_a_cpes: Optional[str] = None
    group_a_definition: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class CISSPAnnualRecommendation(BaseModel):
    recommended_cpes_annually: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class CISSPExamCost(BaseModel):
    exam_cost: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class CISSPInfo(BaseModel):
    experience_requirement: Optional[CISSPExperienceRequirement] = None
    education_waiver: Optional[CISSPEducationWaiver] = None
    renewal: Optional[CISSPRenewal] = None
    group_a: Optional[CISSPGroupA] = None
    annual_recommendation: Optional[CISSPAnnualRecommendation] = None
    exam_cost: Optional[CISSPExamCost] = None


class CombinedExtraction(BaseModel):
    pmp: Optional[PMPInfo] = None
    cissp: Optional[CISSPInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pmp_cissp() -> str:
    return """
    Extract, from the provided answer, all PMP and CISSP requirement details as explicitly stated, along with authoritative reference URLs.
    Return a JSON with two top-level objects: 'pmp' and 'cissp'. For each field, extract values exactly as written in the answer (keep units and wording), and collect all URLs the answer cites for that specific requirement.

    pmp:
      experience:
        months_required: string (e.g., "36 months")
        timeframe_window: string (e.g., "within the last 8 years")
        source_urls: array of URLs
      education:
        contact_hours_required: string (e.g., "35 contact hours")
        source_urls: array of URLs
      renewal:
        total_pdus_required: string (e.g., "60 PDUs")
        cycle_length: string (e.g., "3 years")
        source_urls: array of URLs
      education_minimum:
        min_education_pdus: string (e.g., "35 PDUs")
        source_urls: array of URLs
      talent_triangle:
        min_pdus_per_area: string (e.g., "8 PDUs in each area")
        categories: array of strings (e.g., ["Technical", "Leadership", "Strategic & Business Management"])
        source_urls: array of URLs
      exam_cost:
        member_cost: string (e.g., "$405")
        non_member_cost: string (e.g., "$555")
        source_urls: array of URLs

    cissp:
      experience_requirement:
        min_years: string (e.g., "5 years")
        min_domains_experience: string (e.g., "2 domains")
        total_domains: string (e.g., "8 domains")
        source_urls: array of URLs
      education_waiver:
        waiver_years_with_degree: string (e.g., "1 year")
        source_urls: array of URLs
      renewal:
        total_cpes_required: string (e.g., "120 CPEs")
        cycle_length: string (e.g., "3 years")
        source_urls: array of URLs
      group_a:
        min_group_a_cpes: string (e.g., "90 CPEs")
        group_a_definition: string explaining what Group A credits represent
        source_urls: array of URLs
      annual_recommendation:
        recommended_cpes_annually: string (e.g., "40 CPEs per year")
        source_urls: array of URLs
      exam_cost:
        exam_cost: string (e.g., "$749")
        source_urls: array of URLs

    Rules:
    - Only extract URLs explicitly present in the answer; include all cited URLs per requirement (markdown links are allowed; extract the actual link).
    - If a specific field is not present in the answer, return null for that field or empty list for URLs.
    - Do not infer or invent any values.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def _safe_list_join(items: Optional[List[str]], sep: str = ", ") -> str:
    return sep.join(items) if items else ""


async def _verify_leaf(
    evaluator: Evaluator,
    node_id: str,
    node_desc: str,
    parent,
    claim: str,
    sources: Optional[List[str]],
    exist_gate_node,
    add_ins: str
) -> None:
    leaf = evaluator.add_leaf(id=node_id, desc=node_desc, parent=parent, critical=True)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources if sources else None,
        extra_prerequisites=[exist_gate_node],
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# PMP verification sub-tree                                                   #
# --------------------------------------------------------------------------- #
async def build_pmp_tree(evaluator: Evaluator, parent_node, pmp: Optional[PMPInfo]) -> None:
    pmp_node = evaluator.add_parallel(
        id="PMP_Certification",
        desc="Project Management Professional (PMP) certification requirements",
        parent=parent_node,
        critical=False
    )

    # 1) Experience Requirement
    exp_node = evaluator.add_parallel(
        id="PMP_Experience_Requirement",
        desc="Required months of PM experience for four-year degree holders and the allowed timeframe; include authoritative URL",
        parent=pmp_node,
        critical=True
    )
    exp = pmp.experience if pmp else None
    exp_exists = bool(exp and exp.months_required and exp.timeframe_window and _has_urls(exp.source_urls))
    exp_exist_gate = evaluator.add_custom_node(
        result=exp_exists,
        id="PMP_Experience_Existence",
        desc="PMP experience requirement fields and at least one source URL are provided",
        parent=exp_node,
        critical=True
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_Experience_Duration",
        node_desc="State required months of project management experience for candidates with a four-year degree",
        parent=exp_node,
        claim=f"For PMP candidates with a four-year degree, the required project management experience is {exp.months_required}."
        if exp and exp.months_required else "Required PMP experience months are specified.",
        sources=exp.source_urls if exp else None,
        exist_gate_node=exp_exist_gate,
        add_ins="Verify the exact number of months required; allow wording variations (e.g., 'months' vs 'mo'). Use PMI official pages."
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_Experience_Timeframe",
        node_desc="State the timeframe within which the experience must have been gained",
        parent=exp_node,
        claim=f"The PMP experience must have been obtained {exp.timeframe_window}."
        if exp and exp.timeframe_window else "The PMP experience timeframe is specified.",
        sources=exp.source_urls if exp else None,
        exist_gate_node=exp_exist_gate,
        add_ins="Confirm the timeframe window (e.g., 'within the last X years')."
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_Experience_URL",
        node_desc="Provide authoritative URL documenting the PMP experience requirement",
        parent=exp_node,
        claim=(
            f"PMI's official documentation states the PMP experience requirement: {exp.months_required} "
            f"obtained {exp.timeframe_window} for four-year degree holders."
            if exp else "PMI official documentation states the PMP experience requirement."
        ),
        sources=exp.source_urls if exp else None,
        exist_gate_node=exp_exist_gate,
        add_ins="Ensure at least one official PMI or authoritative URL directly documents this requirement."
    )

    # 2) Education/Contact Hours
    edu_node = evaluator.add_parallel(
        id="PMP_Education_Requirement",
        desc="Required PM education/contact hours before exam; include authoritative URL",
        parent=pmp_node,
        critical=True
    )
    edu = pmp.education if pmp else None
    edu_exists = bool(edu and edu.contact_hours_required and _has_urls(edu.source_urls))
    edu_exist_gate = evaluator.add_custom_node(
        result=edu_exists,
        id="PMP_Education_Existence",
        desc="PMP education/contact-hours field and at least one source URL are provided",
        parent=edu_node,
        critical=True
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_Education_Hours",
        node_desc="State number of contact hours of project management education required before taking the exam",
        parent=edu_node,
        claim=f"PMP requires {edu.contact_hours_required} of project management education before the exam."
        if edu and edu.contact_hours_required else "The PMP contact hour requirement is specified.",
        sources=edu.source_urls if edu else None,
        exist_gate_node=edu_exist_gate,
        add_ins="Verify contact hours requirement; accept 'contact hours' or 'hours of PM education'."
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_Education_URL",
        node_desc="Provide authoritative URL documenting the PMP education/contact-hours requirement",
        parent=edu_node,
        claim=(
            f"The PMP education requirement is {edu.contact_hours_required} contact hours as documented by PMI."
            if edu else "The PMP education requirement is documented by PMI."
        ),
        sources=edu.source_urls if edu else None,
        exist_gate_node=edu_exist_gate,
        add_ins="Ensure the source URL explicitly documents the contact hours requirement."
    )

    # 3) Renewal: Total PDUs and Cycle
    pdu_node = evaluator.add_parallel(
        id="PMP_Renewal_Total_PDUs_and_Cycle",
        desc="Total PDUs required and renewal cycle length; include authoritative URL",
        parent=pmp_node,
        critical=True
    )
    ren = pmp.renewal if pmp else None
    ren_exists = bool(ren and ren.total_pdus_required and ren.cycle_length and _has_urls(ren.source_urls))
    ren_exist_gate = evaluator.add_custom_node(
        result=ren_exists,
        id="PMP_Renewal_Existence",
        desc="PMP renewal total PDUs and cycle fields with at least one source URL are provided",
        parent=pdu_node,
        critical=True
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_PDU_Total_Amount",
        node_desc="State total PDUs required for renewal",
        parent=pdu_node,
        claim=f"PMP renewal requires {ren.total_pdus_required}."
        if ren and ren.total_pdus_required else "The PMP total PDUs for renewal are specified.",
        sources=ren.source_urls if ren else None,
        exist_gate_node=ren_exist_gate,
        add_ins="Verify the total number of PDUs needed for a renewal cycle."
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_PDU_Cycle_Length",
        node_desc="State the renewal cycle length",
        parent=pdu_node,
        claim=f"The PMP renewal cycle length is {ren.cycle_length}."
        if ren and ren.cycle_length else "The PMP renewal cycle length is specified.",
        sources=ren.source_urls if ren else None,
        exist_gate_node=ren_exist_gate,
        add_ins="Verify the length of the renewal cycle (e.g., 'every 3 years')."
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_PDU_Total_URL",
        node_desc="Provide authoritative URL documenting total PDU requirements and cycle length",
        parent=pdu_node,
        claim=(
            f"PMI documentation states PMP renewal requires {ren.total_pdus_required} over a {ren.cycle_length} cycle."
            if ren else "PMI documentation states PMP renewal total PDUs and cycle length."
        ),
        sources=ren.source_urls if ren else None,
        exist_gate_node=ren_exist_gate,
        add_ins="Ensure the URL explicitly shows both PDU total and cycle length."
    )

    # 4) Renewal: Education Minimum
    edu_min_node = evaluator.add_parallel(
        id="PMP_Renewal_Education_Minimum",
        desc="Minimum Education-category PDUs; include authoritative URL",
        parent=pmp_node,
        critical=True
    )
    edu_min = pmp.education_minimum if pmp else None
    edu_min_exists = bool(edu_min and edu_min.min_education_pdus and _has_urls(edu_min.source_urls))
    edu_min_gate = evaluator.add_custom_node(
        result=edu_min_exists,
        id="PMP_Education_Min_Existence",
        desc="PMP Education-category minimum PDUs and at least one source URL are provided",
        parent=edu_min_node,
        critical=True
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_PDU_Education_Amount",
        node_desc="State minimum number of PDUs that must come from the Education category",
        parent=edu_min_node,
        claim=f"At least {edu_min.min_education_pdus} must come from the Education category for PMP renewal."
        if edu_min and edu_min.min_education_pdus else "The minimum Education-category PDUs for PMP renewal are specified.",
        sources=edu_min.source_urls if edu_min else None,
        exist_gate_node=edu_min_gate,
        add_ins="Confirm PMI's Education-category minimum."
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_PDU_Education_URL",
        node_desc="Provide authoritative URL documenting the Education-category minimum",
        parent=edu_min_node,
        claim=(
            f"PMI documentation specifies an Education-category minimum of {edu_min.min_education_pdus} for PMP renewal."
            if edu_min else "PMI documentation specifies an Education-category minimum for PMP renewal."
        ),
        sources=edu_min.source_urls if edu_min else None,
        exist_gate_node=edu_min_gate,
        add_ins="Ensure the URL clearly states the Education-category minimum PDUs requirement."
    )

    # 5) PMI Talent Triangle
    triangle_node = evaluator.add_parallel(
        id="PMP_Talent_Triangle_Requirement",
        desc="Minimum PDUs in each PMI Talent Triangle skill area and list the categories; include authoritative URL",
        parent=pmp_node,
        critical=True
    )
    tri = pmp.talent_triangle if pmp else None
    tri_exists = bool(
        tri and tri.min_pdus_per_area and tri.categories and len(tri.categories) > 0 and _has_urls(tri.source_urls)
    )
    tri_gate = evaluator.add_custom_node(
        result=tri_exists,
        id="PMP_Talent_Triangle_Existence",
        desc="PMI Talent Triangle minimum per area, categories, and at least one source URL are provided",
        parent=triangle_node,
        critical=True
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_Talent_Triangle_Minimum",
        node_desc="State minimum PDUs required in each PMI Talent Triangle skill area",
        parent=triangle_node,
        claim=f"The minimum PDUs required in each PMI Talent Triangle skill area is {tri.min_pdus_per_area}."
        if tri and tri.min_pdus_per_area else "The minimum PDUs per Talent Triangle area are specified.",
        sources=tri.source_urls if tri else None,
        exist_gate_node=tri_gate,
        add_ins="Verify PMI's per-area minimum PDUs requirement."
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_Talent_Triangle_Categories",
        node_desc="List the three PMI Talent Triangle skill area categories",
        parent=triangle_node,
        claim=(
            f"The three PMI Talent Triangle skill area categories are: {_safe_list_join(tri.categories)}."
            if tri else "The three PMI Talent Triangle categories are listed."
        ),
        sources=tri.source_urls if tri else None,
        exist_gate_node=tri_gate,
        add_ins="Confirm the three categories (e.g., Technical, Leadership, Strategic & Business Management). Allow minor naming variations."
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_Talent_Triangle_URL",
        node_desc="Provide authoritative URL documenting the PMI Talent Triangle PDU requirements",
        parent=triangle_node,
        claim=(
            f"PMI documentation specifies {tri.min_pdus_per_area} PDUs in each of the three Talent Triangle areas "
            f"({_safe_list_join(tri.categories)})."
            if tri else "PMI documentation specifies PDU requirements per Talent Triangle area."
        ),
        sources=tri.source_urls if tri else None,
        exist_gate_node=tri_gate,
        add_ins="Ensure the URL explicitly documents both the categories and per-area minimum PDUs."
    )

    # 6) Exam Costs
    cost_node = evaluator.add_parallel(
        id="PMP_Exam_Costs",
        desc="Exam cost for PMI members and non-members; include authoritative URL",
        parent=pmp_node,
        critical=True
    )
    cost = pmp.exam_cost if pmp else None
    cost_exists = bool(cost and cost.member_cost and cost.non_member_cost and _has_urls(cost.source_urls))
    cost_gate = evaluator.add_custom_node(
        result=cost_exists,
        id="PMP_Exam_Costs_Existence",
        desc="PMP exam costs (member & non-member) and at least one source URL are provided",
        parent=cost_node,
        critical=True
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_Member_Exam_Cost",
        node_desc="State PMP exam cost for PMI members",
        parent=cost_node,
        claim=f"The PMP exam cost for PMI members is {cost.member_cost}."
        if cost and cost.member_cost else "The PMP member exam cost is specified.",
        sources=cost.source_urls if cost else None,
        exist_gate_node=cost_gate,
        add_ins="Verify PMI member pricing; accept currency formatting variations."
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_NonMember_Exam_Cost",
        node_desc="State PMP exam cost for non-members",
        parent=cost_node,
        claim=f"The PMP exam cost for non-members is {cost.non_member_cost}."
        if cost and cost.non_member_cost else "The PMP non-member exam cost is specified.",
        sources=cost.source_urls if cost else None,
        exist_gate_node=cost_gate,
        add_ins="Verify non-member pricing; accept currency formatting variations."
    )
    await _verify_leaf(
        evaluator,
        node_id="PMP_Exam_Cost_URL",
        node_desc="Provide authoritative URL documenting PMP exam costs",
        parent=cost_node,
        claim=(
            f"PMI documentation lists PMP exam costs as {cost.member_cost} for members and {cost.non_member_cost} for non-members."
            if cost else "PMI documentation lists PMP exam costs for members and non-members."
        ),
        sources=cost.source_urls if cost else None,
        exist_gate_node=cost_gate,
        add_ins="Ensure the URL is official PMI or an authoritative pricing page explicitly listing the fees."
    )


# --------------------------------------------------------------------------- #
# CISSP verification sub-tree                                                 #
# --------------------------------------------------------------------------- #
async def build_cissp_tree(evaluator: Evaluator, parent_node, cissp: Optional[CISSPInfo]) -> None:
    cissp_node = evaluator.add_parallel(
        id="CISSP_Certification",
        desc="Certified Information Systems Security Professional (CISSP) certification requirements",
        parent=parent_node,
        critical=False
    )

    # 1) Experience requirement
    exp_node = evaluator.add_parallel(
        id="CISSP_Experience_Requirement",
        desc="Minimum years experience, required domain coverage, and total number of domains; include authoritative URL",
        parent=cissp_node,
        critical=True
    )
    exp = cissp.experience_requirement if cissp else None
    exp_exists = bool(
        exp and exp.min_years and exp.min_domains_experience and exp.total_domains and _has_urls(exp.source_urls)
    )
    exp_gate = evaluator.add_custom_node(
        result=exp_exists,
        id="CISSP_Experience_Existence",
        desc="CISSP experience fields (years, min domains, total domains) and at least one source URL are provided",
        parent=exp_node,
        critical=True
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_Experience_Years",
        node_desc="State minimum years of cumulative full-time work experience required",
        parent=exp_node,
        claim=f"The CISSP requires at least {exp.min_years} of cumulative, full-time work experience."
        if exp and exp.min_years else "The CISSP minimum years of experience are specified.",
        sources=exp.source_urls if exp else None,
        exist_gate_node=exp_gate,
        add_ins="Verify minimum full-time experience years; wording variations acceptable."
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_Domain_Coverage",
        node_desc="State minimum number of CISSP domains in which experience is required",
        parent=exp_node,
        claim=f"The CISSP requires experience across at least {exp.min_domains_experience} of its domains."
        if exp and exp.min_domains_experience else "The CISSP minimum number of domains for experience is specified.",
        sources=exp.source_urls if exp else None,
        exist_gate_node=exp_gate,
        add_ins="Verify minimum domain coverage requirement."
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_Total_Domains",
        node_desc="State total number of CISSP domains",
        parent=exp_node,
        claim=f"There are {exp.total_domains} CISSP domains."
        if exp and exp.total_domains else "The total number of CISSP domains is specified.",
        sources=exp.source_urls if exp else None,
        exist_gate_node=exp_gate,
        add_ins="Confirm total number of CISSP domains."
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_Experience_URL",
        node_desc="Provide authoritative URL documenting CISSP experience/domain requirements",
        parent=exp_node,
        claim=(
            f"(ISC)² documentation states CISSP requires {exp.min_years} experience across at least "
            f"{exp.min_domains_experience} of the {exp.total_domains} domains."
            if exp else "(ISC)² documentation states CISSP experience/domain requirements."
        ),
        sources=exp.source_urls if exp else None,
        exist_gate_node=exp_gate,
        add_ins="Ensure the URL clearly shows years required, minimum domains required, and total domains."
    )

    # 2) Education Waiver
    waiver_node = evaluator.add_parallel(
        id="CISSP_Education_Waiver",
        desc="Years of experience that can be waived with a four-year degree; include authoritative URL",
        parent=cissp_node,
        critical=True
    )
    waiver = cissp.education_waiver if cissp else None
    waiver_exists = bool(waiver and waiver.waiver_years_with_degree and _has_urls(waiver.source_urls))
    waiver_gate = evaluator.add_custom_node(
        result=waiver_exists,
        id="CISSP_Waiver_Existence",
        desc="CISSP waiver years with 4-year degree and at least one source URL are provided",
        parent=waiver_node,
        critical=True
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_Waiver_Amount",
        node_desc="State number of years of experience that can be waived with a four-year degree",
        parent=waiver_node,
        claim=f"With a four-year degree, {waiver.waiver_years_with_degree} of experience can be waived for CISSP."
        if waiver and waiver.waiver_years_with_degree else "The CISSP degree-based waiver years are specified.",
        sources=waiver.source_urls if waiver else None,
        exist_gate_node=waiver_gate,
        add_ins="Verify the (ISC)² policy on degree-based experience waiver."
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_Waiver_URL",
        node_desc="Provide authoritative URL documenting the waiver policy",
        parent=waiver_node,
        claim=(
            f"(ISC)² documentation states that {waiver.waiver_years_with_degree} of experience can be waived with a four-year degree."
            if waiver else "(ISC)² documentation states the waiver policy for degree holders."
        ),
        sources=waiver.source_urls if waiver else None,
        exist_gate_node=waiver_gate,
        add_ins="Ensure the URL explicitly documents the degree-based waiver amount."
    )

    # 3) Renewal: Total CPEs and Cycle
    cpe_node = evaluator.add_parallel(
        id="CISSP_Renewal_Total_CPEs_and_Cycle",
        desc="Total CPE credits required and renewal cycle length; include authoritative URL",
        parent=cissp_node,
        critical=True
    )
    cpe = cissp.renewal if cissp else None
    cpe_exists = bool(cpe and cpe.total_cpes_required and cpe.cycle_length and _has_urls(cpe.source_urls))
    cpe_gate = evaluator.add_custom_node(
        result=cpe_exists,
        id="CISSP_Renewal_Existence",
        desc="CISSP renewal total CPEs and cycle fields with at least one source URL are provided",
        parent=cpe_node,
        critical=True
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_CPE_Total_Amount",
        node_desc="State total CPE credits required for renewal",
        parent=cpe_node,
        claim=f"CISSP requires {cpe.total_cpes_required} for renewal."
        if cpe and cpe.total_cpes_required else "The CISSP total CPEs for renewal are specified.",
        sources=cpe.source_urls if cpe else None,
        exist_gate_node=cpe_gate,
        add_ins="Verify the total number of CPE credits required for CISSP renewal."
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_CPE_Cycle_Length",
        node_desc="State the renewal cycle length",
        parent=cpe_node,
        claim=f"The CISSP renewal cycle length is {cpe.cycle_length}."
        if cpe and cpe.cycle_length else "The CISSP renewal cycle length is specified.",
        sources=cpe.source_urls if cpe else None,
        exist_gate_node=cpe_gate,
        add_ins="Verify the length of the renewal cycle (e.g., 'every 3 years')."
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_CPE_Total_URL",
        node_desc="Provide authoritative URL documenting total CPE requirements and cycle length",
        parent=cpe_node,
        claim=(
            f"(ISC)² documentation states CISSP requires {cpe.total_cpes_required} over a {cpe.cycle_length} cycle."
            if cpe else "(ISC)² documentation states CISSP renewal total CPEs and cycle length."
        ),
        sources=cpe.source_urls if cpe else None,
        exist_gate_node=cpe_gate,
        add_ins="Ensure the URL explicitly shows both total CPEs and cycle length."
    )

    # 4) Group A minimum and definition
    group_a_node = evaluator.add_parallel(
        id="CISSP_Group_A_Minimum_and_Definition",
        desc="Minimum Group A CPEs and what Group A represents; include authoritative URL",
        parent=cissp_node,
        critical=True
    )
    ga = cissp.group_a if cissp else None
    ga_exists = bool(ga and ga.min_group_a_cpes and ga.group_a_definition and _has_urls(ga.source_urls))
    ga_gate = evaluator.add_custom_node(
        result=ga_exists,
        id="CISSP_Group_A_Existence",
        desc="CISSP Group A minimum and definition with at least one source URL are provided",
        parent=group_a_node,
        critical=True
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_Group_A_Amount",
        node_desc="State minimum number of CPE credits that must be Group A (domain-related)",
        parent=group_a_node,
        claim=f"At least {ga.min_group_a_cpes} CPEs must be Group A (domain-related) for CISSP."
        if ga and ga.min_group_a_cpes else "The CISSP Group A minimum CPEs are specified.",
        sources=ga.source_urls if ga else None,
        exist_gate_node=ga_gate,
        add_ins="Verify the Group A (domain-related) minimum CPEs for CISSP."
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_Group_A_Definition",
        node_desc="Explain what Group A (domain-related) CPE credits represent",
        parent=group_a_node,
        claim=f"Group A CPE credits are domain-related; specifically: {ga.group_a_definition}."
        if ga and ga.group_a_definition else "Group A CPE credits are domain-related as defined by (ISC)².",
        sources=ga.source_urls if ga else None,
        exist_gate_node=ga_gate,
        add_ins="Confirm the meaning of Group A (domain-related) CPEs as per (ISC)²."
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_Group_A_URL",
        node_desc="Provide authoritative URL documenting Group A requirements/definition",
        parent=group_a_node,
        claim=(
            f"(ISC)² documentation defines Group A CPEs as domain-related and requires at least {ga.min_group_a_cpes}."
            if ga else "(ISC)² documentation defines Group A requirements."
        ),
        sources=ga.source_urls if ga else None,
        exist_gate_node=ga_gate,
        add_ins="Ensure the URL clearly shows Group A definition and minimum requirements."
    )

    # 5) Annual CPE recommendation
    annual_node = evaluator.add_parallel(
        id="CISSP_Annual_CPE_Recommendation",
        desc="Recommended annual CPE earning rate; include authoritative URL",
        parent=cissp_node,
        critical=True
    )
    annual = cissp.annual_recommendation if cissp else None
    annual_exists = bool(annual and annual.recommended_cpes_annually and _has_urls(annual.source_urls))
    annual_gate = evaluator.add_custom_node(
        result=annual_exists,
        id="CISSP_Annual_Existence",
        desc="CISSP recommended annual CPEs and at least one source URL are provided",
        parent=annual_node,
        critical=True
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_Annual_CPE",
        node_desc="State recommended number of CPEs to earn annually",
        parent=annual_node,
        claim=f"It is recommended to earn {annual.recommended_cpes_annually} annually for CISSP."
        if annual and annual.recommended_cpes_annually else "The CISSP annual recommended CPEs are specified.",
        sources=annual.source_urls if annual else None,
        exist_gate_node=annual_gate,
        add_ins="Verify the recommended annual CPE earning rate stated by (ISC)² or authoritative sources."
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_Annual_URL",
        node_desc="Provide authoritative URL documenting annual CPE recommendation",
        parent=annual_node,
        claim=(
            f"(ISC)² or authoritative guidance recommends earning {annual.recommended_cpes_annually} annually."
            if annual else "Authoritative guidance recommends an annual CPE earning rate."
        ),
        sources=annual.source_urls if annual else None,
        exist_gate_node=annual_gate,
        add_ins="Ensure the URL documents the recommended annual number of CPEs."
    )

    # 6) Exam cost
    c_cost_node = evaluator.add_parallel(
        id="CISSP_Exam_Cost",
        desc="CISSP exam cost; include authoritative URL",
        parent=cissp_node,
        critical=True
    )
    c_cost = cissp.exam_cost if cissp else None
    c_cost_exists = bool(c_cost and c_cost.exam_cost and _has_urls(c_cost.source_urls))
    c_cost_gate = evaluator.add_custom_node(
        result=c_cost_exists,
        id="CISSP_Exam_Cost_Existence",
        desc="CISSP exam cost and at least one source URL are provided",
        parent=c_cost_node,
        critical=True
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_Exam_Fee",
        node_desc="State CISSP exam fee",
        parent=c_cost_node,
        claim=f"The CISSP exam fee is {c_cost.exam_cost}."
        if c_cost and c_cost.exam_cost else "The CISSP exam fee is specified.",
        sources=c_cost.source_urls if c_cost else None,
        exist_gate_node=c_cost_gate,
        add_ins="Verify the current CISSP exam fee; accept currency formatting variations."
    )
    await _verify_leaf(
        evaluator,
        node_id="CISSP_Exam_Cost_URL",
        node_desc="Provide authoritative URL documenting CISSP exam cost",
        parent=c_cost_node,
        claim=(
            f"(ISC)² documentation lists the CISSP exam fee as {c_cost.exam_cost}."
            if c_cost else "(ISC)² documentation lists the CISSP exam fee."
        ),
        sources=c_cost.source_urls if c_cost else None,
        exist_gate_node=c_cost_gate,
        add_ins="Ensure the URL is official (ISC)² or authoritative and explicitly lists the fee."
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
    Evaluate an answer for PMP and CISSP requirements with authoritative URLs.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Use parallel; root kept non-critical to allow partial credit across certifications
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

    # Extract combined information
    extracted = await evaluator.extract(
        prompt=prompt_extract_pmp_cissp(),
        template_class=CombinedExtraction,
        extraction_name="combined_pmp_cissp_requirements"
    )

    # Add top-level analysis node reflecting rubric root
    analysis_main = evaluator.add_parallel(
        id="Professional_Certification_Requirements_Analysis",
        desc="Verify and document the complete requirements for PMP and CISSP, including official/authoritative reference URLs for each requirement.",
        parent=root,
        critical=False
    )

    # Build PMP and CISSP subtrees
    await build_pmp_tree(evaluator, analysis_main, extracted.pmp)
    await build_cissp_tree(evaluator, analysis_main, extracted.cissp)

    # Return summary with verification tree
    return evaluator.get_summary()