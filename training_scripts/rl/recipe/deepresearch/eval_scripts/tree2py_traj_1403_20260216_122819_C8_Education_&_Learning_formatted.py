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
TASK_ID = "houston_districts_howard_alignment"
TASK_DESCRIPTION = (
    "A family is relocating to the greater Houston, Texas metropolitan area and is deciding between Fort Bend Independent "
    "School District and Mansfield Independent School District for their current high school sophomore. The student plans "
    "to pursue the Distinguished Level of Achievement graduation plan and apply for Early Action admission to Howard University. "
    "Provide a comprehensive comparison that includes: (1) Local Graduation Requirements - Identify the specific local graduation "
    "requirements that each district imposes beyond the Texas state requirements (including any additional required courses and "
    "total credit counts); (2) Dual Credit Opportunities - For each district, identify the dual credit partner institution and "
    "describe the eligibility requirements or cost structure for dual credit enrollment; (3) Howard University Admission Alignment - "
    "Determine whether each district's graduation requirements satisfy Howard University's recommended high school coursework "
    "(specifically for mathematics, science, and foreign language requirements); (4) Application Timeline - Identify the key deadlines "
    "and requirements for a student targeting Early Action admission to Howard University, including the application deadline, required "
    "recommendation letters, and financial aid submission timing. For each piece of information provided, include reference URL(s) from "
    "official district or university sources that support your answer."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AlignmentItem(BaseModel):
    district_requirement: Optional[str] = None
    district_urls: List[str] = Field(default_factory=list)
    howard_recommendation: Optional[str] = None
    howard_urls: List[str] = Field(default_factory=list)
    conclusion: Optional[str] = None  # Expected values: "meet", "exceed", "not meet" (case-insensitive ok)


class FortBendGrad(BaseModel):
    total_credits: Optional[str] = None
    total_credits_urls: List[str] = Field(default_factory=list)
    health_credit: Optional[str] = None
    health_urls: List[str] = Field(default_factory=list)
    speech_credit: Optional[str] = None  # e.g., Speech/Professional Communications
    speech_urls: List[str] = Field(default_factory=list)


class FortBendDualCredit(BaseModel):
    partner: Optional[str] = None
    partner_urls: List[str] = Field(default_factory=list)
    cost_free_policy: Optional[str] = None  # e.g., "free in HCC taxing district"
    cost_free_urls: List[str] = Field(default_factory=list)
    cost_outside_policy: Optional[str] = None  # e.g., "$65 per course outside"
    cost_outside_urls: List[str] = Field(default_factory=list)


class FortBendAlignment(BaseModel):
    math: Optional[AlignmentItem] = None
    science: Optional[AlignmentItem] = None
    foreign_language: Optional[AlignmentItem] = None


class MISDGrad(BaseModel):
    total_credits: Optional[str] = None
    total_credits_urls: List[str] = Field(default_factory=list)
    prof_comm_credit: Optional[str] = None  # "Professional Communications" 0.5 credit
    prof_comm_urls: List[str] = Field(default_factory=list)
    health_credit: Optional[str] = None
    health_urls: List[str] = Field(default_factory=list)
    science_spec_text: Optional[str] = None  # "4 science credits including Biology and either Chemistry or Physics, plus 2 additional science credits"
    science_spec_urls: List[str] = Field(default_factory=list)
    lote_credits: Optional[str] = None  # "2 credits in same language"
    lote_urls: List[str] = Field(default_factory=list)


class MISDDualCredit(BaseModel):
    partner: Optional[str] = None
    partner_urls: List[str] = Field(default_factory=list)
    eligibility_gpa: Optional[str] = None  # "80+ GPA"
    eligibility_gpa_urls: List[str] = Field(default_factory=list)
    eligibility_tsi: Optional[str] = None  # "passing TSI"
    eligibility_tsi_urls: List[str] = Field(default_factory=list)


class MISDAlignment(BaseModel):
    math: Optional[AlignmentItem] = None
    science: Optional[AlignmentItem] = None
    foreign_language: Optional[AlignmentItem] = None


class HowardTimeline(BaseModel):
    early_action_deadline: Optional[str] = None
    deadline_urls: List[str] = Field(default_factory=list)
    recommendation_letters: Optional[str] = None  # "two letters: one counselor, one teacher"
    rec_urls: List[str] = Field(default_factory=list)
    financial_aid_timing: Optional[str] = None  # "FAFSA by application deadline for priority scholarship consideration"
    fa_urls: List[str] = Field(default_factory=list)


class ComparisonExtraction(BaseModel):
    fb_grad: Optional[FortBendGrad] = None
    fb_dual: Optional[FortBendDualCredit] = None
    fb_align: Optional[FortBendAlignment] = None
    misd_grad: Optional[MISDGrad] = None
    misd_dual: Optional[MISDDualCredit] = None
    misd_align: Optional[MISDAlignment] = None
    howard_timeline: Optional[HowardTimeline] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_comparison() -> str:
    return """
Extract the following structured information exactly as presented in the answer. For every claim, also extract the official supporting URL(s) listed in the answer (district or university pages). If a field is not present, return null (or empty list for URLs).

Fort Bend ISD (FBISD) – Local Graduation Requirements:
- fb_grad.total_credits: total credits required to graduate with an endorsement and Distinguished Level of Achievement (DLA)
- fb_grad.total_credits_urls: official FBISD URL(s) supporting total credits
- fb_grad.health_credit: the Health requirement (e.g., "0.5 credit")
- fb_grad.health_urls: official FBISD URL(s) supporting Health requirement
- fb_grad.speech_credit: the Speech/Professional Communications requirement (e.g., "0.5 credit")
- fb_grad.speech_urls: official FBISD URL(s) supporting Speech requirement

Fort Bend ISD – Dual Credit:
- fb_dual.partner: the dual credit partner institution (e.g., "Houston Community College" or "HCC")
- fb_dual.partner_urls: official FBISD/HCC URL(s) supporting the partner
- fb_dual.cost_free_policy: policy description if free within HCC taxing district (as stated in the answer)
- fb_dual.cost_free_urls: official FBISD/HCC URL(s) supporting free in-district policy
- fb_dual.cost_outside_policy: policy description for students outside HCC taxing district (e.g., "$65 per course")
- fb_dual.cost_outside_urls: official FBISD/HCC URL(s) supporting out-of-district cost

Fort Bend ISD – Alignment to Howard (DLA context):
Each of fb_align.{math,science,foreign_language} should include:
- district_requirement: the applicable graduation requirement for that subject (as stated in the answer)
- district_urls: official FBISD/Texas source URL(s) supporting the district requirement
- howard_recommendation: Howard’s recommended high school coursework for that subject (as stated)
- howard_urls: official Howard URL(s) supporting the recommendation
- conclusion: the answer’s conclusion for alignment ("meet", "exceed", or "not meet")

Mansfield ISD (MISD) – Local Graduation Requirements:
- misd_grad.total_credits: total credits required to graduate with endorsement
- misd_grad.total_credits_urls: official MISD URL(s) supporting total credits
- misd_grad.prof_comm_credit: Professional Communications requirement (e.g., "0.5 credit")
- misd_grad.prof_comm_urls: official MISD URL(s) supporting Professional Communications requirement
- misd_grad.health_credit: Health requirement (e.g., "0.5 credit")
- misd_grad.health_urls: official MISD URL(s) supporting Health requirement
- misd_grad.science_spec_text: science requirement specification (e.g., "4 science credits including Biology and either Chemistry or Physics, plus 2 additional science credits")
- misd_grad.science_spec_urls: official MISD URL(s) supporting science requirement spec
- misd_grad.lote_credits: LOTE requirement (e.g., "2 credits in the same language")
- misd_grad.lote_urls: official MISD URL(s) supporting LOTE requirement

Mansfield ISD – Dual Credit:
- misd_dual.partner: the dual credit partner institution (e.g., "Tarrant County College" or "TCC")
- misd_dual.partner_urls: official MISD/TCC URL(s) supporting the partner
- misd_dual.eligibility_gpa: GPA requirement for dual credit (e.g., "80+ GPA")
- misd_dual.eligibility_gpa_urls: official MISD URL(s) supporting GPA requirement
- misd_dual.eligibility_tsi: TSI requirement for dual credit (e.g., "passing TSI")
- misd_dual.eligibility_tsi_urls: official MISD URL(s) supporting TSI requirement

Mansfield ISD – Alignment to Howard (DLA context):
Each of misd_align.{math,science,foreign_language} should include:
- district_requirement
- district_urls
- howard_recommendation
- howard_urls
- conclusion

Howard University Early Action Timeline:
- howard_timeline.early_action_deadline: the EA deadline (e.g., "November 15")
- howard_timeline.deadline_urls: official Howard URL(s) supporting deadline
- howard_timeline.recommendation_letters: the requirement for letters (e.g., "two letters: one from counselor and one from teacher")
- howard_timeline.rec_urls: official Howard URL(s) supporting letters requirement
- howard_timeline.financial_aid_timing: FAFSA/financial aid timing for priority scholarship consideration (e.g., "by application deadline")
- howard_timeline.fa_urls: official Howard URL(s) supporting FAFSA timing

Rules:
- Only extract URLs explicitly mentioned in the answer text (valid HTTP/HTTPS links).
- Do not invent or infer URLs.
- Preserve the subject names and requirement descriptions as phrased in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper functions for verification                                           #
# --------------------------------------------------------------------------- #
def _has_nonempty_urls(urls: Optional[List[str]]) -> bool:
    return isinstance(urls, list) and any(isinstance(u, str) and u.strip() for u in urls)


async def verify_claim_with_urls(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    urls: Optional[List[str]],
    critical: bool = True,
    additional_instruction: Optional[str] = None
) -> bool:
    """
    Create a leaf and verify a claim against provided URL(s).
    If no URLs or URLs are empty, directly fail the leaf to enforce source-grounding.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical
    )

    if not _has_nonempty_urls(urls):
        # Fail due to missing sources
        leaf.score = 0.0
        leaf.status = "failed"
        return False

    return await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=additional_instruction or "None"
    )


async def verify_alignment_section(
    evaluator: Evaluator,
    parent_node,
    section_id: str,
    section_desc: str,
    align_item: Optional[AlignmentItem],
    subject_name: str
) -> None:
    """
    For each subject alignment (math/science/foreign language), verify three parts:
    1) District requirement supported by district URLs
    2) Howard recommendation supported by Howard URLs
    3) The conclusion (meet/exceed/not meet) is logically consistent with the requirements
    """
    # Create a parallel node to house the three critical checks
    section_node = evaluator.add_parallel(
        id=section_id,
        desc=section_desc,
        parent=parent_node,
        critical=True
    )

    # Guard extracted availability
    district_req = align_item.district_requirement if align_item else None
    district_urls = align_item.district_urls if align_item else []
    howard_rec = align_item.howard_recommendation if align_item else None
    howard_urls = align_item.howard_urls if align_item else []
    conclusion = (align_item.conclusion or "").strip().lower() if align_item and align_item.conclusion else ""

    # 1) District requirement supported
    await verify_claim_with_urls(
        evaluator,
        section_node,
        f"{section_id}_district_req_supported",
        f"{subject_name} district requirement is correctly stated and supported",
        claim=f"According to the cited district source(s), the {subject_name} graduation requirement for the student's plan is: {district_req}.",
        urls=district_urls,
        critical=True,
        additional_instruction=f"Verify that the webpage explicitly supports the stated {subject_name} requirement for graduation under the applicable plan (DLA/FHSP with endorsement). Allow minor phrasing differences."
    )

    # 2) Howard recommendation supported
    await verify_claim_with_urls(
        evaluator,
        section_node,
        f"{section_id}_howard_rec_supported",
        f"{subject_name} Howard recommendation is correctly stated and supported",
        claim=f"Howard University recommends the following {subject_name} high school coursework: {howard_rec}.",
        urls=howard_urls,
        critical=True,
        additional_instruction=f"Verify this is stated on an official Howard University admissions page. Allow minor phrasing differences (e.g., 'years'/'units', etc.)."
    )

    # 3) Conclusion validity (simple verify without URLs - logical check only)
    concl_leaf = evaluator.add_leaf(
        id=f"{section_id}_conclusion_valid",
        desc=f"{subject_name} alignment conclusion is logically valid (meet/exceed/not meet) given district requirement and Howard recommendation",
        parent=section_node,
        critical=True
    )
    # Provide detailed instruction for logical interpretation
    logic_instruction = (
        "Interpret credits roughly as years (1.0 credit ≈ 1 year). "
        "Rule of thumb for alignment: if the district requirement years ≥ Howard's recommended years → 'meet' or 'exceed' is acceptable "
        "(use 'exceed' only if strictly greater); if less → 'not meet'. "
        "If recommendations include 'with lab', ensure the district requirement reasonably includes lab sciences if applicable. "
        "Allow reasonable curriculum naming variations (e.g., 'LOTE' for foreign language)."
    )
    concl_claim = (
        f"Given the district requirement '{district_req}' and Howard's recommendation '{howard_rec}', "
        f"the answer's conclusion that this {subject_name} requirement '{conclusion}' Howard's recommendation is logically correct."
    )
    await evaluator.verify(
        claim=concl_claim,
        node=concl_leaf,
        sources=None,  # Logical check only
        additional_instruction=logic_instruction
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_fort_bend_nodes(evaluator: Evaluator, parent_node, data: ComparisonExtraction) -> None:
    fb_node = evaluator.add_parallel(
        id="Fort_Bend_ISD",
        desc="Fort Bend ISD required components with official citations.",
        parent=parent_node,
        critical=True
    )

    fb_grad = data.fb_grad or FortBendGrad()
    fb_dual = data.fb_dual or FortBendDualCredit()
    fb_align = data.fb_align or FortBendAlignment()

    # FB Local Grad Total Credits (expects 26 if correctly stated in answer)
    await verify_claim_with_urls(
        evaluator,
        fb_node,
        "FB_Local_Grad_Total_Credits",
        "States Fort Bend ISD total credits required for graduation with endorsement and Distinguished Level of Achievement and cites an official FBISD URL",
        claim=f"Fort Bend ISD requires a total of {fb_grad.total_credits} credits to graduate with an endorsement and the Distinguished Level of Achievement (DLA).",
        urls=fb_grad.total_credits_urls,
        critical=True,
        additional_instruction="Verify the page states total credits for graduation with endorsement and DLA. Allow minor wording differences."
    )

    # FB Local Grad Health (0.5 credit expected if stated)
    await verify_claim_with_urls(
        evaluator,
        fb_node,
        "FB_Local_Grad_Health",
        "States Fort Bend ISD local Health requirement and cites an official FBISD URL",
        claim=f"Fort Bend ISD requires {fb_grad.health_credit} credit in Health for graduation.",
        urls=fb_grad.health_urls,
        critical=True,
        additional_instruction="Verify that the graduation requirements explicitly include the stated Health credit amount."
    )

    # FB Local Grad Speech (0.5 credit expected if stated)
    await verify_claim_with_urls(
        evaluator,
        fb_node,
        "FB_Local_Grad_Speech",
        "States Fort Bend ISD local Speech/Professional Communications requirement and cites an official FBISD URL",
        claim=f"Fort Bend ISD requires {fb_grad.speech_credit} credit in Speech/Professional Communications for graduation.",
        urls=fb_grad.speech_urls,
        critical=True,
        additional_instruction="Verify that the graduation requirements explicitly include the stated Speech/Professional Communications credit amount. Allow synonyms."
    )

    # FB Dual Credit Partner (HCC)
    await verify_claim_with_urls(
        evaluator,
        fb_node,
        "FB_Dual_Credit_Partner",
        "Identifies Fort Bend ISD dual credit partner institution and cites an official FBISD/HCC URL",
        claim=f"Fort Bend ISD's dual credit partner institution is {fb_dual.partner}.",
        urls=fb_dual.partner_urls,
        critical=True,
        additional_instruction="Verify that the cited page states the dual credit partner. Accept 'HCC' as 'Houston Community College'."
    )

    # FB Dual Credit Cost: split into two critical leaves under a critical sub-node
    cost_node = evaluator.add_parallel(
        id="FB_Dual_Credit_Cost",
        desc="Describes Fort Bend ISD dual credit cost structure with official citations.",
        parent=fb_node,
        critical=True
    )

    await verify_claim_with_urls(
        evaluator,
        cost_node,
        "FB_Dual_Credit_Cost_Free_In_District",
        "States FBISD dual credit is free for students in the HCC taxing district with official citation",
        claim=f"Fort Bend ISD dual credit tuition is free for students residing within the HCC taxing district as stated: {fb_dual.cost_free_policy}.",
        urls=fb_dual.cost_free_urls,
        critical=True,
        additional_instruction="Verify the page states no tuition/tuition-free for students in the HCC taxing district. Allow synonymous phrases like 'no cost' or 'at no charge'."
    )

    await verify_claim_with_urls(
        evaluator,
        cost_node,
        "FB_Dual_Credit_Cost_Outside_District",
        "States FBISD dual credit cost for students outside the HCC taxing district with official citation",
        claim=f"For students outside the HCC taxing district, Fort Bend ISD dual credit cost is as stated: {fb_dual.cost_outside_policy}.",
        urls=fb_dual.cost_outside_urls,
        critical=True,
        additional_instruction="Verify the page states the specific out-of-district cost (e.g., a dollar amount per course). Allow minor formatting differences like '$65' vs '65 dollars'."
    )

    # FB Alignment: Math, Science, Foreign Language
    if fb_align and fb_align.math:
        await verify_alignment_section(
            evaluator,
            fb_node,
            "FB_Howard_Alignment_Math",
            "Fort Bend ISD math alignment to Howard's recommendation with citations and valid conclusion",
            fb_align.math,
            "mathematics"
        )
    else:
        # Create failing children if missing
        missing_node = evaluator.add_leaf(
            id="FB_Howard_Alignment_Math_missing",
            desc="Fort Bend ISD math alignment section missing in answer",
            parent=fb_node,
            critical=True
        )
        missing_node.score = 0.0
        missing_node.status = "failed"

    if fb_align and fb_align.science:
        await verify_alignment_section(
            evaluator,
            fb_node,
            "FB_Howard_Alignment_Science",
            "Fort Bend ISD science alignment to Howard's recommendation with citations and valid conclusion",
            fb_align.science,
            "science"
        )
    else:
        missing_node = evaluator.add_leaf(
            id="FB_Howard_Alignment_Science_missing",
            desc="Fort Bend ISD science alignment section missing in answer",
            parent=fb_node,
            critical=True
        )
        missing_node.score = 0.0
        missing_node.status = "failed"

    if fb_align and fb_align.foreign_language:
        await verify_alignment_section(
            evaluator,
            fb_node,
            "FB_Howard_Alignment_Foreign_Language",
            "Fort Bend ISD foreign language alignment to Howard's recommendation with citations and valid conclusion",
            fb_align.foreign_language,
            "foreign language"
        )
    else:
        missing_node = evaluator.add_leaf(
            id="FB_Howard_Alignment_Foreign_Language_missing",
            desc="Fort Bend ISD foreign language alignment section missing in answer",
            parent=fb_node,
            critical=True
        )
        missing_node.score = 0.0
        missing_node.status = "failed"


async def build_mansfield_nodes(evaluator: Evaluator, parent_node, data: ComparisonExtraction) -> None:
    misd_node = evaluator.add_parallel(
        id="Mansfield_ISD",
        desc="Mansfield ISD required components with official citations.",
        parent=parent_node,
        critical=True
    )

    misd_grad = data.misd_grad or MISDGrad()
    misd_dual = data.misd_dual or MISDDualCredit()
    misd_align = data.misd_align or MISDAlignment()

    # MISD Local Grad Total Credits (expects 26 if correctly stated)
    await verify_claim_with_urls(
        evaluator,
        misd_node,
        "MISD_Local_Grad_Total_Credits",
        "States Mansfield ISD total credits required for graduation with endorsement and cites an official MISD URL",
        claim=f"Mansfield ISD requires a total of {misd_grad.total_credits} credits to graduate with an endorsement.",
        urls=misd_grad.total_credits_urls,
        critical=True,
        additional_instruction="Verify the page states total credits for graduation with endorsement (DLA context acceptable). Allow minor wording differences."
    )

    # MISD Local Grad Professional Communications (0.5 credit)
    await verify_claim_with_urls(
        evaluator,
        misd_node,
        "MISD_Local_Grad_Professional_Communications",
        "States Mansfield ISD Professional Communications requirement and cites an official MISD URL",
        claim=f"Mansfield ISD requires {misd_grad.prof_comm_credit} credit in Professional Communications for graduation.",
        urls=misd_grad.prof_comm_urls,
        critical=True,
        additional_instruction="Verify that the graduation requirements explicitly include the Professional Communications credit amount."
    )

    # MISD Local Grad Health (0.5 credit)
    await verify_claim_with_urls(
        evaluator,
        misd_node,
        "MISD_Local_Grad_Health",
        "States Mansfield ISD Health requirement and cites an official MISD URL",
        claim=f"Mansfield ISD requires {misd_grad.health_credit} credit in Health for graduation.",
        urls=misd_grad.health_urls,
        critical=True,
        additional_instruction="Verify that the graduation requirements explicitly include the Health credit amount."
    )

    # MISD Local Grad Science Spec
    await verify_claim_with_urls(
        evaluator,
        misd_node,
        "MISD_Local_Grad_Science_Spec",
        "States Mansfield ISD science graduation requirement specification and cites an official MISD URL",
        claim=f"Mansfield ISD science requirement is stated as: {misd_grad.science_spec_text}.",
        urls=misd_grad.science_spec_urls,
        critical=True,
        additional_instruction="Verify that the page states the science specification (e.g., Biology + either Chemistry or Physics, plus additional science credits) consistent with the claim."
    )

    # MISD Local Grad LOTE
    await verify_claim_with_urls(
        evaluator,
        misd_node,
        "MISD_Local_Grad_LOTE",
        "States Mansfield ISD LOTE requirement and cites an official MISD URL",
        claim=f"Mansfield ISD requires {misd_grad.lote_credits} in the same language (LOTE) for graduation.",
        urls=misd_grad.lote_urls,
        critical=True,
        additional_instruction="Verify that the page states the LOTE requirement, typically two credits in the same language."
    )

    # MISD Dual Credit Partner (TCC)
    await verify_claim_with_urls(
        evaluator,
        misd_node,
        "MISD_Dual_Credit_Partner",
        "Identifies Mansfield ISD dual credit partner institution and cites an official MISD/TCC URL",
        claim=f"Mansfield ISD's dual credit partner institution is {misd_dual.partner}.",
        urls=misd_dual.partner_urls,
        critical=True,
        additional_instruction="Verify that the cited page identifies Tarrant County College (TCC) as the partner. Allow 'TCC' synonym."
    )

    # MISD Dual Credit Eligibility GPA (80+)
    await verify_claim_with_urls(
        evaluator,
        misd_node,
        "MISD_Dual_Credit_Eligibility_GPA",
        "States Mansfield ISD dual credit eligibility GPA requirement and cites an official MISD URL",
        claim=f"Mansfield ISD requires {misd_dual.eligibility_gpa} for dual credit eligibility.",
        urls=misd_dual.eligibility_gpa_urls,
        critical=True,
        additional_instruction="Verify the page states the GPA threshold for dual credit eligibility (e.g., 80+ GPA)."
    )

    # MISD Dual Credit Eligibility TSI (passing)
    await verify_claim_with_urls(
        evaluator,
        misd_node,
        "MISD_Dual_Credit_Eligibility_TSI",
        "States Mansfield ISD dual credit eligibility TSI requirement and cites an official MISD URL",
        claim=f"Mansfield ISD requires {misd_dual.eligibility_tsi} for dual credit eligibility.",
        urls=misd_dual.eligibility_tsi_urls,
        critical=True,
        additional_instruction="Verify the page states that a passing TSI (Texas Success Initiative) status is required for dual credit."
    )

    # MISD Alignment: Math, Science, Foreign Language
    if misd_align and misd_align.math:
        await verify_alignment_section(
            evaluator,
            misd_node,
            "MISD_Howard_Alignment_Math",
            "Mansfield ISD math alignment to Howard's recommendation with citations and valid conclusion",
            misd_align.math,
            "mathematics"
        )
    else:
        missing_node = evaluator.add_leaf(
            id="MISD_Howard_Alignment_Math_missing",
            desc="Mansfield ISD math alignment section missing in answer",
            parent=misd_node,
            critical=True
        )
        missing_node.score = 0.0
        missing_node.status = "failed"

    if misd_align and misd_align.science:
        await verify_alignment_section(
            evaluator,
            misd_node,
            "MISD_Howard_Alignment_Science",
            "Mansfield ISD science alignment to Howard's recommendation with citations and valid conclusion",
            misd_align.science,
            "science"
        )
    else:
        missing_node = evaluator.add_leaf(
            id="MISD_Howard_Alignment_Science_missing",
            desc="Mansfield ISD science alignment section missing in answer",
            parent=misd_node,
            critical=True
        )
        missing_node.score = 0.0
        missing_node.status = "failed"

    if misd_align and misd_align.foreign_language:
        await verify_alignment_section(
            evaluator,
            misd_node,
            "MISD_Howard_Alignment_Foreign_Language",
            "Mansfield ISD foreign language alignment to Howard's recommendation with citations and valid conclusion",
            misd_align.foreign_language,
            "foreign language"
        )
    else:
        missing_node = evaluator.add_leaf(
            id="MISD_Howard_Alignment_Foreign_Language_missing",
            desc="Mansfield ISD foreign language alignment section missing in answer",
            parent=misd_node,
            critical=True
        )
        missing_node.score = 0.0
        missing_node.status = "failed"


async def build_howard_timeline_nodes(evaluator: Evaluator, parent_node, data: ComparisonExtraction) -> None:
    hu_node = evaluator.add_parallel(
        id="Howard_University_Early_Action_Timeline",
        desc="Howard University Early Action timeline items requested (deadline, recommendation letters, and financial aid timing), each with official citations.",
        parent=parent_node,
        critical=True
    )

    ht = data.howard_timeline or HowardTimeline()

    # Early Action Deadline (e.g., Nov 15)
    await verify_claim_with_urls(
        evaluator,
        hu_node,
        "Howard_Early_Action_Deadline",
        "States Howard University Early Action application deadline and cites an official Howard URL",
        claim=f"Howard University's Early Action application deadline is {ht.early_action_deadline}.",
        urls=ht.deadline_urls,
        critical=True,
        additional_instruction="Verify the official Howard admissions site states the Early Action deadline (e.g., November 15)."
    )

    # Recommendation Letters (two letters: one counselor, one teacher)
    await verify_claim_with_urls(
        evaluator,
        hu_node,
        "Howard_Recommendation_Letters",
        "States Howard requires two letters of recommendation (one from counselor, one from teacher) with official citation",
        claim=f"Howard University requires the following recommendation letters: {ht.recommendation_letters}.",
        urls=ht.rec_urls,
        critical=True,
        additional_instruction="Verify that Howard specifies two recommendations including one from a counselor and one from a teacher (or equivalent phrasing)."
    )

    # Financial Aid Timing (FAFSA by application deadline for priority scholarship consideration)
    await verify_claim_with_urls(
        evaluator,
        hu_node,
        "Howard_Financial_Aid_Timing",
        "States FAFSA submission timing for priority scholarship consideration and cites an official Howard URL",
        claim=f"For priority scholarship consideration, FAFSA timing is as stated: {ht.financial_aid_timing}.",
        urls=ht.fa_urls,
        critical=True,
        additional_instruction="Verify that Howard indicates FAFSA/financial aid materials should be submitted by the application deadline (or the specified timing) for priority scholarship consideration."
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
    Evaluate an answer for the Fort Bend ISD vs Mansfield ISD vs Howard EA alignment task.
    """
    # Initialize evaluator with a parallel root (children critical to enforce gating)
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

    # Extract structured comparison info
    extracted = await evaluator.extract(
        prompt=prompt_extract_comparison(),
        template_class=ComparisonExtraction,
        extraction_name="comparison_extraction"
    )

    # Build and verify Fort Bend ISD node
    await build_fort_bend_nodes(evaluator, root, extracted)

    # Build and verify Mansfield ISD node
    await build_mansfield_nodes(evaluator, root, extracted)

    # Build and verify Howard Early Action timeline node
    await build_howard_timeline_nodes(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()