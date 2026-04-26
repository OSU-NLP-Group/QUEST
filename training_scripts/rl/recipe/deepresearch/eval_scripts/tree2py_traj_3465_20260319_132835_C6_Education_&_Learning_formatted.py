import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "relocation_pa_to_va_2025_2026"
TASK_DESCRIPTION = """A family is relocating from Philadelphia, Pennsylvania to Loudoun County, Virginia in summer 2026. Their high school student will be entering 11th grade and aims to pursue the most rigorous diploma option available in their new school district. The parents want to understand key differences in graduation requirements and prepare to participate in local school governance.

Compile a comprehensive relocation information package that includes:

Part 1 - Graduation Credit Requirements Comparison:
Compare the total credit requirements and subject-specific credit distributions between:
- The School District of Philadelphia's standard graduation requirements
- Loudoun County Public Schools' Virginia Advanced Studies Diploma requirements

For each district, provide:
- Total credits required (for Virginia, specify both standard credits and verified credits separately)
- Credit requirements for: English, Mathematics, Science, Social Studies/History, and World Language

Include URL references to official district sources for these credit requirements.

Part 2 - Additional Graduation Requirements:
Identify the additional non-credit requirements (beyond course credits) that students must complete to graduate in each district, including:
- For Philadelphia: Project requirements and pathway/assessment requirements
- For Virginia Advanced Studies Diploma: All additional competency-based, experiential, or skills-based requirements beyond earning credits

Include URL references to official sources for these additional requirements.

Part 3 - Administrative Leadership:
Provide the full name and title of the current superintendent for both:
- The School District of Philadelphia
- Loudoun County Public Schools

Include the first day of school for the 2025-2026 academic year for both districts, and the last day of school for LCPS 2025-2026.

Include URL references to official sources for this administrative information.

Part 4 - LCPS Board Meeting Participation:
The parents want to attend and potentially speak at Loudoun County Public Schools board meetings. Provide:
- The categories of individuals who are eligible to speak at LCPS School Board meetings
- The acceptable forms of documentation/proof required to demonstrate eligibility
- The complete registration timeline (when pre-registration opens before the meeting, when it closes, and the policy for walk-up registration)

Include URL reference to the official LCPS policy on citizen participation.

All information must be current as of the 2025-2026 school year. Provide specific, factual details with proper source attribution.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PhillyCredits(BaseModel):
    total_credits: Optional[str] = None
    english: Optional[str] = None
    mathematics: Optional[str] = None
    science: Optional[str] = None
    social_studies_history: Optional[str] = None
    world_language: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class VirginiaAdvancedCredits(BaseModel):
    standard_credits_total: Optional[str] = None
    verified_credits_total: Optional[str] = None
    english: Optional[str] = None
    mathematics: Optional[str] = None
    science: Optional[str] = None
    social_studies_history: Optional[str] = None
    world_language: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PhillyAdditionalRequirements(BaseModel):
    project_requirement: Optional[str] = None
    pathway_assessment_requirement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class VirginiaAdditionalRequirements(BaseModel):
    ap_honors_ib_dual_or_wbl_or_cte: Optional[str] = None
    virtual_course: Optional[str] = None
    first_aid_cpr_aed: Optional[str] = None
    five_cs: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PhillyAdmin(BaseModel):
    superintendent_name: Optional[str] = None
    superintendent_title: Optional[str] = None
    first_day_2025_2026: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LcpsAdmin(BaseModel):
    superintendent_name: Optional[str] = None
    superintendent_title: Optional[str] = None
    first_day_2025_2026: Optional[str] = None
    last_day_2025_2026: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LcpsBoardParticipation(BaseModel):
    policy_url: Optional[str] = None
    eligibility_categories: List[str] = Field(default_factory=list)
    acceptable_proof: List[str] = Field(default_factory=list)
    prereg_opens: Optional[str] = None
    prereg_closes: Optional[str] = None
    walkup_policy: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RelocationPackageExtraction(BaseModel):
    philly_credits: Optional[PhillyCredits] = None
    va_advanced_credits: Optional[VirginiaAdvancedCredits] = None
    philly_additional: Optional[PhillyAdditionalRequirements] = None
    va_additional: Optional[VirginiaAdditionalRequirements] = None
    admin_philly: Optional[PhillyAdmin] = None
    admin_lcps: Optional[LcpsAdmin] = None
    lcps_board: Optional[LcpsBoardParticipation] = None
    comparison_present: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_relocation_package() -> str:
    return """
    Extract the requested structured information from the answer. Return null for any field not present. Preserve wording and numbers as stated in the answer (do not infer). Extract only URLs explicitly shown in the answer.

    1) philly_credits (School District of Philadelphia standard diploma):
       - total_credits: total credits required to graduate
       - english: credit requirement for English
       - mathematics: credit requirement for Mathematics
       - science: credit requirement for Science
       - social_studies_history: credit requirement for Social Studies/History
       - world_language: credit requirement for World Language
       - sources: list of official URLs cited for these credit requirements (district/state official pages only if present)

    2) va_advanced_credits (Virginia Advanced Studies Diploma as applicable in LCPS):
       - standard_credits_total: total standard credits required
       - verified_credits_total: total verified credits required
       - english: credit requirement for English
       - mathematics: credit requirement for Mathematics
       - science: credit requirement for Science (state if laboratory sciences if indicated)
       - social_studies_history: credit requirement for Social Studies/History (a.k.a. History/Social Sciences)
       - world_language: credit requirement for World Language
       - sources: list of official URLs cited for these credit requirements (LCPS or VDOE official pages only if present)

    3) philly_additional (Philadelphia non-credit graduation requirements):
       - project_requirement: Multidisciplinary Project or Service Learning Project (state exactly as described), if provided
       - pathway_assessment_requirement: describe the PDE Pathways to Graduation requirement mentioned, if provided
       - sources: official URLs supporting these non-credit requirements

    4) va_additional (Virginia non-credit requirements for the Advanced Studies Diploma):
       - ap_honors_ib_dual_or_wbl_or_cte: statement that students must complete AP/honors/IB/dual enrollment OR a high-quality WBL experience OR an approved CTE credential (verbatim if provided)
       - virtual_course: statement about the one virtual course requirement
       - first_aid_cpr_aed: statement about First Aid/CPR/AED training, including waiver condition if mentioned
       - five_cs: statement about demonstrating the 5 Cs (critical thinking, creative thinking, collaboration, communication, citizenship)
       - sources: official URLs supporting these requirements (LCPS or VDOE)

    5) admin_philly (School District of Philadelphia leadership and calendar):
       - superintendent_name
       - superintendent_title
       - first_day_2025_2026: first day of school for 2025-2026
       - sources: official URLs for superintendent info and calendar date

    6) admin_lcps (Loudoun County Public Schools leadership and calendar):
       - superintendent_name
       - superintendent_title
       - first_day_2025_2026
       - last_day_2025_2026
       - sources: official URLs for superintendent info and calendar dates

    7) lcps_board (LCPS School Board public comment/citizen participation):
       - policy_url: the single official policy/regulation page URL cited (if available)
       - eligibility_categories: list categories eligible to speak
       - acceptable_proof: list acceptable documentation/proof forms
       - prereg_opens: when pre-registration opens (day/time, relative timing)
       - prereg_closes: when pre-registration closes (time on meeting day)
       - walkup_policy: walk-up registration policy (cutoff and placement)
       - sources: list of any URLs cited for these items (include policy_url if present)

    8) comparison_present:
       - Set to true if the answer clearly provides an explicit comparison (e.g., side-by-side table or described differences) of totals and the five subject-area credit requirements between the two districts; otherwise false or null.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _val(v: Optional[str]) -> str:
    return v if v is not None else ""

def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification: Part 1 - Credit Requirements                                  #
# --------------------------------------------------------------------------- #
async def verify_part_1(evaluator: Evaluator, parent_node, data: RelocationPackageExtraction) -> None:
    part_node = evaluator.add_parallel(
        id="Part_1_Credit_Requirements_Comparison",
        desc="Compare graduation credit requirements between Philadelphia and LCPS/VA Advanced Studies Diploma",
        parent=parent_node,
        critical=True
    )

    ph = data.philly_credits or PhillyCredits()
    va = data.va_advanced_credits or VirginiaAdvancedCredits()

    # Official URLs for BOTH districts' credit requirement info
    leaf_official_urls = evaluator.add_leaf(
        id="Official_URLs_Credit_Requirements",
        desc="Provide official-source URL references for BOTH districts' credit requirement information",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim="These URLs are official district or state sources that state graduation credit requirements for the School District of Philadelphia and/or for the Virginia Advanced Studies Diploma used by LCPS.",
        node=leaf_official_urls,
        sources=_merge_sources(ph.sources, va.sources),
        additional_instruction="Accept official pages such as philasd.org, education.pa.gov, lcps.org, or doe.virginia.gov. The page must explicitly describe graduation credit requirements."
    )

    # Philadelphia requirements (Totals and 5 subjects)
    leaf_ph_total = evaluator.add_leaf(
        id="Philadelphia_Total_Credits",
        desc="State the total credits required to graduate in the School District of Philadelphia",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The School District of Philadelphia requires a total of '{_val(ph.total_credits)}' credits to graduate.",
        node=leaf_ph_total,
        sources=ph.sources,
        additional_instruction="Verify the total number of credits required to graduate as stated on an official Philadelphia source."
    )

    leaf_ph_eng = evaluator.add_leaf(
        id="Philadelphia_English_Credits",
        desc="State Philadelphia credit requirement for English",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Philadelphia requires '{_val(ph.english)}' English credits for graduation.",
        node=leaf_ph_eng,
        sources=ph.sources,
        additional_instruction="Confirm the English credit requirement on an official Philadelphia source."
    )

    leaf_ph_math = evaluator.add_leaf(
        id="Philadelphia_Mathematics_Credits",
        desc="State Philadelphia credit requirement for Mathematics",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Philadelphia requires '{_val(ph.mathematics)}' Mathematics credits for graduation.",
        node=leaf_ph_math,
        sources=ph.sources,
        additional_instruction="Confirm the Mathematics credit requirement on an official Philadelphia source."
    )

    leaf_ph_sci = evaluator.add_leaf(
        id="Philadelphia_Science_Credits",
        desc="State Philadelphia credit requirement for Science",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Philadelphia requires '{_val(ph.science)}' Science credits for graduation.",
        node=leaf_ph_sci,
        sources=ph.sources,
        additional_instruction="Confirm the Science credit requirement on an official Philadelphia source."
    )

    leaf_ph_hist = evaluator.add_leaf(
        id="Philadelphia_Social_Studies_History_Credits",
        desc="State Philadelphia credit requirement for Social Studies/History",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Philadelphia requires '{_val(ph.social_studies_history)}' Social Studies/History credits for graduation.",
        node=leaf_ph_hist,
        sources=ph.sources,
        additional_instruction="Confirm the Social Studies/History credit requirement on an official Philadelphia source."
    )

    leaf_ph_wl = evaluator.add_leaf(
        id="Philadelphia_World_Language_Credits",
        desc="State Philadelphia credit requirement for World Language",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Philadelphia requires '{_val(ph.world_language)}' World Language credits (or equivalent policy) for graduation.",
        node=leaf_ph_wl,
        sources=ph.sources,
        additional_instruction="Confirm any World Language credit expectation or explicit statement about World Language requirements on an official Philadelphia source."
    )

    # Virginia Advanced Studies (Totals and 5 subjects)
    leaf_va_totals = evaluator.add_leaf(
        id="Virginia_Advanced_Studies_Total_Credits",
        desc="State Virginia Advanced Studies Diploma total required standard credits AND verified credits separately",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For the Virginia Advanced Studies Diploma, total required standard credits are '{_val(va.standard_credits_total)}' and total required verified credits are '{_val(va.verified_credits_total)}'.",
        node=leaf_va_totals,
        sources=va.sources,
        additional_instruction="Confirm both total standard credits and total verified credits required for the Advanced Studies Diploma using LCPS or VDOE official pages."
    )

    leaf_va_eng = evaluator.add_leaf(
        id="Virginia_English_Credits",
        desc="State Virginia Advanced Studies Diploma credit requirement for English",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Virginia Advanced Studies Diploma requires '{_val(va.english)}' English credits.",
        node=leaf_va_eng,
        sources=va.sources,
        additional_instruction="Confirm the English credit requirement for the Advanced Studies Diploma on LCPS or VDOE official pages."
    )

    leaf_va_math = evaluator.add_leaf(
        id="Virginia_Mathematics_Credits",
        desc="State Virginia Advanced Studies Diploma credit requirement for Mathematics",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Virginia Advanced Studies Diploma requires '{_val(va.mathematics)}' Mathematics credits.",
        node=leaf_va_math,
        sources=va.sources,
        additional_instruction="Confirm the Mathematics credit requirement for the Advanced Studies Diploma on LCPS or VDOE official pages."
    )

    leaf_va_sci = evaluator.add_leaf(
        id="Virginia_Science_Credits",
        desc="State Virginia Advanced Studies Diploma credit requirement for Science (Laboratory Science if applicable)",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Virginia Advanced Studies Diploma requires '{_val(va.science)}' Science credits.",
        node=leaf_va_sci,
        sources=va.sources,
        additional_instruction="Confirm the Science credit requirement (and lab specification if indicated) on LCPS or VDOE official pages."
    )

    leaf_va_hist = evaluator.add_leaf(
        id="Virginia_Social_Studies_History_Credits",
        desc="State Virginia Advanced Studies Diploma credit requirement for Social Studies/History (History/Social Sciences if applicable)",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Virginia Advanced Studies Diploma requires '{_val(va.social_studies_history)}' History/Social Sciences credits.",
        node=leaf_va_hist,
        sources=va.sources,
        additional_instruction="Confirm the History/Social Sciences credit requirement on LCPS or VDOE official pages."
    )

    leaf_va_wl = evaluator.add_leaf(
        id="Virginia_World_Language_Credits",
        desc="State Virginia Advanced Studies Diploma credit requirement for World Language",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Virginia Advanced Studies Diploma requires '{_val(va.world_language)}' World Language credits (or an approved sequence).",
        node=leaf_va_wl,
        sources=va.sources,
        additional_instruction="Confirm the World Language requirement (e.g., 3 years of one language or 2+2 of two languages) on LCPS or VDOE pages."
    )

    # Explicit comparison leaf
    leaf_compare = evaluator.add_leaf(
        id="Explicit_Comparison",
        desc="Provide an explicit comparison (e.g., side-by-side table and/or described differences) of totals and the five subject-area credit requirements between the two districts",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer includes an explicit comparison (such as a side-by-side table or clearly described differences) of totals and the five subject-area credit requirements between the School District of Philadelphia and the Virginia Advanced Studies Diploma (LCPS).",
        node=leaf_compare,
        additional_instruction="Check the answer text for an explicit comparison. It may be a table, bullet list, or a clearly structured side-by-side narrative."
    )


# --------------------------------------------------------------------------- #
# Verification: Part 2 - Additional Graduation Requirements                   #
# --------------------------------------------------------------------------- #
async def verify_part_2(evaluator: Evaluator, parent_node, data: RelocationPackageExtraction) -> None:
    part_node = evaluator.add_parallel(
        id="Part_2_Additional_Graduation_Requirements",
        desc="Non-credit graduation requirements for Philadelphia and Virginia Advanced Studies Diploma",
        parent=parent_node,
        critical=True
    )

    ph_add = data.philly_additional or PhillyAdditionalRequirements()
    va_add = data.va_additional or VirginiaAdditionalRequirements()

    # Official URLs for BOTH districts' additional requirements
    leaf_urls = evaluator.add_leaf(
        id="Official_URLs_Additional_Requirements",
        desc="Provide official-source URL references for BOTH districts' additional (non-credit) graduation requirements",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim="These URLs are official district or state sources that state non-credit graduation requirements for the School District of Philadelphia and/or for the Virginia Advanced Studies Diploma used by LCPS.",
        node=leaf_urls,
        sources=_merge_sources(ph_add.sources, va_add.sources),
        additional_instruction="Accept official pages such as philasd.org, education.pa.gov, lcps.org, or doe.virginia.gov."
    )

    # Philadelphia: Project requirement
    leaf_ph_project = evaluator.add_leaf(
        id="Philadelphia_Project_Requirement",
        desc="Identify Philadelphia's project requirement option (Multidisciplinary Project OR Service Learning Project)",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Philadelphia requires: '{_val(ph_add.project_requirement)}' as a graduation project option.",
        node=leaf_ph_project,
        sources=ph_add.sources,
        additional_instruction="Confirm the described project requirement from the official School District of Philadelphia or PA sources."
    )

    # Philadelphia: Pathway/assessment requirement
    leaf_ph_pathway = evaluator.add_leaf(
        id="Philadelphia_Pathway_Assessment_Requirement",
        desc="Identify Philadelphia's pathway/assessment requirement to satisfy one of the PDE Pathways to Graduation",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Philadelphia requires meeting a pathway/assessment requirement to satisfy one of the PDE Pathways to Graduation: '{_val(ph_add.pathway_assessment_requirement)}'.",
        node=leaf_ph_pathway,
        sources=ph_add.sources,
        additional_instruction="Confirm the requirement referencing PDE Pathways to Graduation on official School District of Philadelphia or PA Department of Education sources."
    )

    # Virginia: AP/honors/IB/dual OR WBL OR CTE
    leaf_va_combo = evaluator.add_leaf(
        id="Virginia_AP_Honors_IB_DualEnroll_OR_WBL_OR_CTE",
        desc="Identify Virginia requirement to complete AP/honors/IB/dual enrollment OR a high-quality work-based learning experience OR an approved CTE credential",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim="For the Virginia Advanced Studies Diploma, students must complete either an AP/honors/IB/dual enrollment course OR a high-quality work-based learning experience OR an approved CTE credential.",
        node=leaf_va_combo,
        sources=va_add.sources,
        additional_instruction="Verify this non-credit graduation requirement on LCPS or VDOE official documentation."
    )

    # Virginia: Virtual course
    leaf_va_virtual = evaluator.add_leaf(
        id="Virginia_Virtual_Course",
        desc="Identify Virginia requirement to successfully complete one virtual course",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim="For the Virginia Advanced Studies Diploma, students must successfully complete one virtual course.",
        node=leaf_va_virtual,
        sources=va_add.sources,
        additional_instruction="Confirm the one virtual course requirement on LCPS or VDOE official pages."
    )

    # Virginia: First Aid, CPR, AED
    leaf_va_cpr = evaluator.add_leaf(
        id="Virginia_FirstAid_CPR_AED",
        desc="Identify Virginia requirement for First Aid, CPR, and AED training (including waiver condition if applicable)",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For the Virginia Advanced Studies Diploma, students must complete training in First Aid, CPR, and use of an AED. Details stated: '{_val(va_add.first_aid_cpr_aed)}'.",
        node=leaf_va_cpr,
        sources=va_add.sources,
        additional_instruction="Confirm the CPR/First Aid/AED training requirement and any waiver conditions on LCPS or VDOE official pages."
    )

    # Virginia: Five Cs
    leaf_va_fivecs = evaluator.add_leaf(
        id="Virginia_Five_Cs",
        desc="Identify Virginia requirement to demonstrate the 5 C's (critical thinking, creative thinking, collaboration, communication, citizenship)",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim="Virginia requires students to demonstrate the 5 Cs: critical thinking, creative thinking, collaboration, communication, and citizenship.",
        node=leaf_va_fivecs,
        sources=va_add.sources,
        additional_instruction="Confirm the 5 Cs requirement on LCPS or VDOE official pages."
    )


# --------------------------------------------------------------------------- #
# Verification: Part 3 - Administrative Leadership and Calendar               #
# --------------------------------------------------------------------------- #
async def verify_part_3(evaluator: Evaluator, parent_node, data: RelocationPackageExtraction) -> None:
    part_node = evaluator.add_parallel(
        id="Part_3_Administrative_Leadership_and_Calendar",
        desc="Superintendent names/titles and required 2025-2026 calendar dates for both districts (and last day for LCPS)",
        parent=parent_node,
        critical=True
    )

    ph_ad = data.admin_philly or PhillyAdmin()
    lcps_ad = data.admin_lcps or LcpsAdmin()

    # Official URLs covering superintendent info and calendar dates
    leaf_admin_urls = evaluator.add_leaf(
        id="Official_URLs_Admin_Info",
        desc="Provide official-source URL references covering superintendent information AND the required 2025-2026 calendar dates",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim="These URLs are official district sources that provide superintendent information and/or 2025-2026 school calendar dates for Philadelphia and LCPS.",
        node=leaf_admin_urls,
        sources=_merge_sources(ph_ad.sources, lcps_ad.sources),
        additional_instruction="Accept only official district pages (philasd.org for Philadelphia, lcps.org for Loudoun). The page must include superintendent info or calendar dates."
    )

    # Philadelphia superintendent
    leaf_ph_sup = evaluator.add_leaf(
        id="Philadelphia_Superintendent",
        desc="Provide the full name and title of the current School District of Philadelphia superintendent",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The School District of Philadelphia superintendent is '{_val(ph_ad.superintendent_name)}' with the title '{_val(ph_ad.superintendent_title)}'.",
        node=leaf_ph_sup,
        sources=ph_ad.sources,
        additional_instruction="Verify current superintendent name and title on official School District of Philadelphia sources."
    )

    # LCPS superintendent
    leaf_lcps_sup = evaluator.add_leaf(
        id="LCPS_Superintendent",
        desc="Provide the full name and title of the current Loudoun County Public Schools superintendent",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Loudoun County Public Schools superintendent is '{_val(lcps_ad.superintendent_name)}' with the title '{_val(lcps_ad.superintendent_title)}'.",
        node=leaf_lcps_sup,
        sources=lcps_ad.sources,
        additional_instruction="Verify current superintendent name and title on official LCPS sources."
    )

    # First/Last day dates
    leaf_ph_first = evaluator.add_leaf(
        id="Philadelphia_First_Day_2025_2026",
        desc="Provide the first day of school for Philadelphia for the 2025-2026 academic year",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For the 2025-2026 school year, the first day of school for the School District of Philadelphia is '{_val(ph_ad.first_day_2025_2026)}'.",
        node=leaf_ph_first,
        sources=ph_ad.sources,
        additional_instruction="Check an official Philadelphia district calendar or announcement page for the 2025-2026 first day of school."
    )

    leaf_lcps_first = evaluator.add_leaf(
        id="LCPS_First_Day_2025_2026",
        desc="Provide the first day of school for LCPS for the 2025-2026 academic year",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For the 2025-2026 school year, the first day of school for LCPS is '{_val(lcps_ad.first_day_2025_2026)}'.",
        node=leaf_lcps_first,
        sources=lcps_ad.sources,
        additional_instruction="Check an official LCPS district calendar or announcement page for the 2025-2026 first day of school."
    )

    leaf_lcps_last = evaluator.add_leaf(
        id="LCPS_Last_Day_2025_2026",
        desc="Provide the last day of school for LCPS for the 2025-2026 academic year",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For the 2025-2026 school year, the last day of school for LCPS is '{_val(lcps_ad.last_day_2025_2026)}'.",
        node=leaf_lcps_last,
        sources=lcps_ad.sources,
        additional_instruction="Check an official LCPS district calendar or announcement page for the 2025-2026 last day of school."
    )


# --------------------------------------------------------------------------- #
# Verification: Part 4 - LCPS Board Meeting Participation                     #
# --------------------------------------------------------------------------- #
async def verify_part_4(evaluator: Evaluator, parent_node, data: RelocationPackageExtraction) -> None:
    part_node = evaluator.add_parallel(
        id="Part_4_LCPS_Board_Meeting_Participation",
        desc="LCPS School Board meeting speaking eligibility, proof, and registration timeline per official policy",
        parent=parent_node,
        critical=True
    )

    board = data.lcps_board or LcpsBoardParticipation()
    all_board_sources = _merge_sources([board.policy_url] if board.policy_url else [], board.sources)

    # Official policy URL
    leaf_policy = evaluator.add_leaf(
        id="Official_URL_Citizen_Participation_Policy",
        desc="Provide an official-source URL reference to the LCPS policy on citizen participation/public comment",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL is the official LCPS policy/regulation page for public comment or citizen participation at School Board meetings.",
        node=leaf_policy,
        sources=board.policy_url if board.policy_url else all_board_sources,
        additional_instruction="Accept only lcps.org or documents hosted by LCPS as official policy/regulation pages."
    )

    # Eligibility categories
    leaf_elig = evaluator.add_leaf(
        id="Eligibility_Categories",
        desc="List the categories of individuals who are eligible to speak at LCPS School Board meetings",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The following are eligible categories to speak at LCPS School Board meetings: {board.eligibility_categories}.",
        node=leaf_elig,
        sources=all_board_sources,
        additional_instruction="Confirm these categories appear in the LCPS citizen participation/public comment policy."
    )

    # Acceptable proof
    leaf_proof = evaluator.add_leaf(
        id="Proof_Documentation",
        desc="Identify the acceptable forms of documentation/proof required to demonstrate eligibility",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Acceptable documentation/proof for eligibility includes: {board.acceptable_proof}.",
        node=leaf_proof,
        sources=all_board_sources,
        additional_instruction="Confirm the documentation/proof forms directly match the LCPS policy."
    )

    # Pre-registration opens
    leaf_open = evaluator.add_leaf(
        id="PreRegistration_Opens",
        desc="State when pre-registration opens before the meeting (day and time)",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Pre-registration opens: '{_val(board.prereg_opens)}'.",
        node=leaf_open,
        sources=all_board_sources,
        additional_instruction="Confirm the open time/day for pre-registration exactly as stated in the LCPS policy."
    )

    # Pre-registration closes
    leaf_close = evaluator.add_leaf(
        id="PreRegistration_Closes",
        desc="State when pre-registration closes (time on meeting day)",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Pre-registration closes: '{_val(board.prereg_closes)}'.",
        node=leaf_close,
        sources=all_board_sources,
        additional_instruction="Confirm the closing time for pre-registration on the meeting day as stated in the LCPS policy."
    )

    # Walk-up policy
    leaf_walk = evaluator.add_leaf(
        id="WalkUp_Registration_Policy",
        desc="State the walk-up registration policy including cutoff time and how walk-ups are placed in speaker order",
        parent=part_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Walk-up registration policy: '{_val(board.walkup_policy)}'.",
        node=leaf_walk,
        sources=all_board_sources,
        additional_instruction="Confirm the stated walk-up registration process, cutoff, and ordering rules in the LCPS policy."
    )


# --------------------------------------------------------------------------- #
# Verification: Currency As Of 2025-2026                                      #
# --------------------------------------------------------------------------- #
async def verify_currency(evaluator: Evaluator, parent_node, data: RelocationPackageExtraction) -> None:
    leaf = evaluator.add_leaf(
        id="Currency_AsOf_2025_2026",
        desc="All provided facts are current/applicable for the 2025-2026 school year (sources or statements clearly indicate 2025-2026 applicability)",
        parent=parent_node,
        critical=True
    )
    ph_ad = data.admin_philly or PhillyAdmin()
    lcps_ad = data.admin_lcps or LcpsAdmin()
    board = data.lcps_board or LcpsBoardParticipation()

    # Use admin calendar sources (most likely to explicitly show 2025-2026), plus policy if mentioned with year tags
    sources = _merge_sources(ph_ad.sources, lcps_ad.sources, [board.policy_url] if board.policy_url else [])
    await evaluator.verify(
        claim="The sources include official 2025-2026 school calendar or dated policy pages indicating that the information is current for the 2025–2026 school year.",
        node=leaf,
        sources=sources,
        additional_instruction="Look for explicit '2025-2026', '2025–26', or equivalent school-year labeling on official district pages, especially calendars."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is parallel per rubric
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

    # Root is CRITICAL per rubric; all children under a critical parent must also be critical
    root.critical = True

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_relocation_package(),
        template_class=RelocationPackageExtraction,
        extraction_name="relocation_package_extraction"
    )

    # Build verification tree according to rubric
    # 1) Currency leaf
    await verify_currency(evaluator, root, extracted)

    # 2) Part 1 - Credit requirements
    await verify_part_1(evaluator, root, extracted)

    # 3) Part 2 - Additional non-credit requirements
    await verify_part_2(evaluator, root, extracted)

    # 4) Part 3 - Admin leadership and calendar
    await verify_part_3(evaluator, root, extracted)

    # 5) Part 4 - LCPS Board meeting participation
    await verify_part_4(evaluator, root, extracted)

    # Optional: record ground-truth expectations summary/context (not strict GT)
    evaluator.add_custom_info(
        info={
            "notes": "All checks are grounded to official district/state sources where applicable; currency emphasizes 2025-2026 calendars.",
            "parts": ["Part 1: Credits", "Part 2: Additional requirements", "Part 3: Admin+Calendar", "Part 4: Board participation"]
        },
        info_type="eval_notes",
        info_name="evaluation_notes"
    )

    return evaluator.get_summary()