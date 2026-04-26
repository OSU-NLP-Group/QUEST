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
TASK_ID = "msu_nacoe_four_programs"
TASK_DESCRIPTION = """
Identify four distinct ABET-accredited undergraduate engineering programs offered by Montana State University's Norm Asbjornson College of Engineering. For each program, provide the following information:

1. Program Name
2. ABET Accreditation (Engineering Accreditation Commission of ABET)
3. College Affiliation (Norm Asbjornson College of Engineering at MSU)
4. Credit Requirements:
   - Minimum total number of credits required for graduation
   - Confirmation that the program requires a minimum of 42 credits in courses numbered 300 and above
5. Capstone Design Requirement:
   - Confirmation that the program requires a capstone design experience
   - The specific course code(s) for the capstone sequence
   - The total number of credit hours for the capstone sequence
6. Fundamentals of Engineering Exam:
   - Confirmation that the program requires students to take the FE Exam (typically EGEN 488)
7. Professional Electives Requirements:
   - Minimum number of professional elective credits
   - Any additional constraints on professional electives
8. Reference URL: Official MSU academic catalog page documenting these requirements

Ensure that all four programs you identify are distinct from one another, and that all information is verifiable through official MSU sources.
"""

# Program-specific expected details (used when rubric expects specific values)
EXPECTED_CONFIG = {
    "mechanical": {
        "expected_name": "Mechanical Engineering",
        "total_credits": "128",
        "capstone_codes": ["EMEC 489R", "EMEC 499R"],
        "prof_electives_min": "12",
    },
    "civil": {
        "expected_name": "Civil Engineering",
        "total_credits": "123",
        "capstone_codes": ["ECIV 499R"],
        "prof_electives_min": "15",
    },
    "chemical": {
        "expected_name": "Chemical Engineering",
        "total_credits": "122",
        "capstone_codes": ["ECHM 411R", "ECHM 412R"],
        "prof_electives_min": None,  # The rubric only requires "a specific numeric value is stated" (not a fixed target)
    },
    "electrical": {
        "expected_name": "Electrical Engineering",
        "total_credits": "125",
        "capstone_codes": ["EELE 488R", "EELE 489R"],
        "prof_electives_min": "27",
    },
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramDetails(BaseModel):
    program_name: Optional[str] = None
    source_url: Optional[str] = None
    # Text snippets captured from the answer (not strictly enforced, used to create claims)
    abet_accreditation_text: Optional[str] = None
    college_affiliation_text: Optional[str] = None
    total_credits_min: Optional[str] = None
    upper_division_42_text: Optional[str] = None
    capstone_required_text: Optional[str] = None
    capstone_course_codes: List[str] = Field(default_factory=list)
    capstone_total_credits: Optional[str] = None
    fe_exam_required_text: Optional[str] = None
    fe_exam_reg_egen_488_text: Optional[str] = None
    fe_exam_final_semester_text: Optional[str] = None
    prof_electives_min_credits: Optional[str] = None
    prof_electives_constraints_text: Optional[str] = None
    four_year_plan_text: Optional[str] = None


class FourProgramsExtraction(BaseModel):
    mechanical: Optional[ProgramDetails] = None
    civil: Optional[ProgramDetails] = None
    chemical: Optional[ProgramDetails] = None
    electrical: Optional[ProgramDetails] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_four_programs() -> str:
    return """
Extract the requested details for four specific undergraduate engineering programs at Montana State University, as stated in the answer:
Programs to extract (use these keys): mechanical, civil, chemical, electrical.

For each of the four programs, extract the following fields:
- program_name: The program's official name as provided in the answer.
- source_url: The URL to the official MSU academic catalog page that documents the program requirements.
- abet_accreditation_text: Any text indicating ABET EAC accreditation as provided in the answer.
- college_affiliation_text: Any text indicating the program is housed in the Norm Asbjornson College of Engineering.
- total_credits_min: The minimum total credits for graduation (as stated in the answer).
- upper_division_42_text: Any text indicating the program requires at least 42 credits in 300-level and above courses.
- capstone_required_text: Any text indicating a capstone design experience is required.
- capstone_course_codes: A list of the specific capstone course code(s) mentioned (e.g., ["EMEC 489R","EMEC 499R"]).
- capstone_total_credits: The total number of credits for the capstone sequence as stated in the answer (e.g., "5", "6").
- fe_exam_required_text: Any text indicating the FE exam is required.
- fe_exam_reg_egen_488_text: Any text indicating the FE exam is registered as EGEN 488 (0 credits).
- fe_exam_final_semester_text: Any text indicating the FE exam is required in the final semester.
- prof_electives_min_credits: The minimum number of professional elective credits required, as stated in the answer.
- prof_electives_constraints_text: Any additional constraints on professional electives (e.g., design-intensive requirements, limits on certain credit categories).
- four_year_plan_text: Any text indicating a documented four-year semester-by-semester plan.

Rules:
- Extract exactly what the answer states. If a field is not present, use null (or an empty list for capstone_course_codes).
- source_url must be the official MSU academic catalog page for the program (if present in the answer). If the answer gives multiple links, pick the most relevant catalog page.
Return a JSON object with keys: mechanical, civil, chemical, electrical, each containing the fields above (or null if that program is missing in the answer).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_name(s: Optional[str]) -> str:
    if not s:
        return ""
    return "".join(ch.lower() for ch in s if ch.isalnum())


async def _verify_leaf_by_url(
    evaluator: Evaluator,
    *,
    leaf_id: str,
    desc: str,
    parent,
    url: Optional[str],
    claim: str,
    critical: bool = True,
    add_ins: Optional[str] = None,
) -> None:
    node = evaluator.add_leaf(id=leaf_id, desc=desc, parent=parent, critical=critical)
    base_ins = (
        "Only mark Correct if this exact fact is explicitly supported by the provided official Montana State University "
        "academic catalog page for this program. If the URL is missing, unrelated, or not accessible, mark Incorrect."
    )
    if add_ins:
        base_ins = base_ins + " " + add_ins
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=url,
        additional_instruction=base_ins,
    )


async def _verify_leaf_simple(
    evaluator: Evaluator,
    *,
    leaf_id: str,
    desc: str,
    parent,
    claim: str,
    critical: bool = True,
    add_ins: Optional[str] = None,
) -> None:
    node = evaluator.add_leaf(id=leaf_id, desc=desc, parent=parent, critical=critical)
    instruction = add_ins or "Use reasonable fuzzy matching for names/titles; ignore case and minor formatting."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=None,
        additional_instruction=instruction,
    )


def _capstone_codes_text(codes: List[str]) -> str:
    if not codes:
        return ""
    if len(codes) == 1:
        return codes[0]
    return " and ".join([", ".join(codes[:-1]), codes[-1]]) if len(codes) > 2 else " and ".join(codes)


# --------------------------------------------------------------------------- #
# Program-specific verification routines                                      #
# --------------------------------------------------------------------------- #
async def verify_mechanical(evaluator: Evaluator, parent_node, prog: Optional[ProgramDetails]) -> None:
    cfg = EXPECTED_CONFIG["mechanical"]
    url = prog.source_url if prog else None
    me_node = evaluator.add_parallel(
        id="Mechanical_Engineering",
        desc="Mechanical Engineering program requirements and documentation",
        parent=parent_node,
        critical=False,
    )

    # ME_Name (simple match against expected)
    provided_name = prog.program_name if prog else ""
    await _verify_leaf_simple(
        evaluator,
        leaf_id="ME_Name",
        desc="Program name is Mechanical Engineering",
        parent=me_node,
        claim=f"The provided program name for Mechanical is equivalent to 'Mechanical Engineering'. Provided: '{provided_name}'.",
        critical=True,
        add_ins="Treat variations like 'Mechanical Engineering BS' or 'B.S. in Mechanical Engineering' as equivalent. If no name is provided, mark Incorrect.",
    )

    # ABET EAC
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ME_ABET_EAC",
        desc="Program is accredited by ABET Engineering Accreditation Commission (EAC)",
        parent=me_node,
        url=url,
        claim="This catalog page states the program is accredited by the Engineering Accreditation Commission (EAC) of ABET.",
        critical=True,
    )

    # College Affiliation
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ME_College_Affiliation",
        desc="Program is housed in MSU's Norm Asbjornson College of Engineering",
        parent=me_node,
        url=url,
        claim="This catalog page indicates the program is housed within the Norm Asbjornson College of Engineering at Montana State University.",
        critical=True,
    )

    # Total Credits = 128
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ME_Total_Credits_128",
        desc="Minimum total credits required for graduation is 128",
        parent=me_node,
        url=url,
        claim="The minimum total number of credits required for graduation in this program is 128.",
        critical=True,
    )

    # Upper Division 42
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ME_Upper_Division_42",
        desc="Requires a minimum of 42 credits in courses numbered 300 and above",
        parent=me_node,
        url=url,
        claim="The program requires at least 42 credits in courses numbered 300 and above.",
        critical=True,
    )

    # Capstone required
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ME_Capstone_Required",
        desc="Requires a capstone design experience",
        parent=me_node,
        url=url,
        claim="This program requires a capstone design experience (senior design).",
        critical=True,
    )

    # Capstone course codes
    expected_codes_text = _capstone_codes_text(cfg["capstone_codes"])
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ME_Capstone_Courses_EMEC_489R_499R",
        desc="Capstone course code(s) are EMEC 489R and EMEC 499R",
        parent=me_node,
        url=url,
        claim=f"The specific capstone course codes are {expected_codes_text}.",
        critical=True,
    )

    # Capstone total credits verified (uses answer's number if provided)
    capstone_total = (prog.capstone_total_credits or "").strip() if prog else ""
    capstone_claim = (
        f"The total number of credit hours for the capstone sequence ({expected_codes_text}) is {capstone_total} in total."
        if capstone_total else
        "The catalog page explicitly states the total number of credit hours for the capstone sequence for this program."
    )
    add_ins = "If the answer did not provide a specific total, or the number does not match the catalog, mark Incorrect."
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ME_Capstone_Total_Credits_Verified",
        desc="States the total capstone sequence credit hours and it matches the cited official MSU catalog source",
        parent=me_node,
        url=url,
        claim=capstone_claim,
        critical=True,
        add_ins=add_ins,
    )

    # FE Exam required
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ME_FE_Exam_Required",
        desc="Requires students to take the Fundamentals of Engineering (FE) Exam",
        parent=me_node,
        url=url,
        claim="The program requires students to take the Fundamentals of Engineering (FE) exam.",
        critical=True,
    )

    # FE Exam registered as EGEN 488 (0 credits)
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ME_FE_Exam_Registered_As_EGEN_488_0cr",
        desc="FE Exam is typically registered as EGEN 488 (0 credits)",
        parent=me_node,
        url=url,
        claim="The FE exam is typically registered as EGEN 488 and carries 0 credits.",
        critical=True,
    )

    # FE Exam final semester
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ME_FE_Exam_Final_Semester",
        desc="FE Exam is required during the final semester",
        parent=me_node,
        url=url,
        claim="The FE exam must be taken during the program's final semester.",
        critical=True,
    )

    # Professional Electives minimum credits = 12
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ME_Professional_Electives_Min_12",
        desc="Minimum professional elective credits required is 12",
        parent=me_node,
        url=url,
        claim="The minimum number of professional elective credits required for this program is 12.",
        critical=True,
    )

    # Professional Electives additional constraints documented (non-critical)
    constraints_text = prog.prof_electives_constraints_text if prog else None
    constraints_claim = (
        f"The answer documents the additional constraints on professional electives for this program as: '{constraints_text}', "
        "and these constraints are supported by the catalog page. If the catalog has no additional constraints, stating 'none' is acceptable."
    )
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ME_Professional_Electives_Additional_Constraints_Documented",
        desc="Any additional professional elective constraints are documented, or explicitly stated as none",
        parent=me_node,
        url=url,
        claim=constraints_claim,
        critical=False,
        add_ins="If the answer omits constraints while the catalog lists them, mark Incorrect.",
    )

    # Four-year semester-by-semester plan
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ME_Four_Year_Semester_By_Semester_Plan",
        desc="Program has a documented semester-by-semester curriculum structure spanning four academic years",
        parent=me_node,
        url=url,
        claim="The catalog page includes a semester-by-semester curriculum plan spanning four academic years.",
        critical=True,
        add_ins="Accept 'four-year plan' or 'curriculum by semester' or similarly named sections.",
    )

    # Source URL presence (simple check against the answer)
    await _verify_leaf_simple(
        evaluator,
        leaf_id="ME_Source_URL",
        desc="Provides an official MSU academic catalog URL documenting these requirements",
        parent=me_node,
        claim=f"The answer provides an official MSU academic catalog URL for the Mechanical Engineering program: {url}.",
        critical=True,
        add_ins="The URL should be from the official MSU catalog (e.g., catalog.montana.edu). If no URL is provided, mark Incorrect.",
    )


async def verify_civil(evaluator: Evaluator, parent_node, prog: Optional[ProgramDetails]) -> None:
    cfg = EXPECTED_CONFIG["civil"]
    url = prog.source_url if prog else None
    ce_node = evaluator.add_parallel(
        id="Civil_Engineering",
        desc="Civil Engineering program requirements and documentation",
        parent=parent_node,
        critical=False,
    )

    # CE_Name
    provided_name = prog.program_name if prog else ""
    await _verify_leaf_simple(
        evaluator,
        leaf_id="CE_Name",
        desc="Program name is Civil Engineering",
        parent=ce_node,
        claim=f"The provided program name for Civil is equivalent to 'Civil Engineering'. Provided: '{provided_name}'.",
        critical=True,
        add_ins="Treat variations like 'Civil Engineering BS' as equivalent. If no name is provided, mark Incorrect.",
    )

    # ABET EAC
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="CE_ABET_EAC",
        desc="Program is accredited by ABET Engineering Accreditation Commission (EAC)",
        parent=ce_node,
        url=url,
        claim="This catalog page states the program is accredited by the Engineering Accreditation Commission (EAC) of ABET.",
        critical=True,
    )

    # College Affiliation
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="CE_College_Affiliation",
        desc="Program is housed in MSU's Norm Asbjornson College of Engineering",
        parent=ce_node,
        url=url,
        claim="This catalog page indicates the program is housed within the Norm Asbjornson College of Engineering at Montana State University.",
        critical=True,
    )

    # Total Credits = 123
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="CE_Total_Credits_123",
        desc="Minimum total credits required for graduation is 123",
        parent=ce_node,
        url=url,
        claim="The minimum total number of credits required for graduation in this program is 123.",
        critical=True,
    )

    # Upper Division 42
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="CE_Upper_Division_42",
        desc="Requires a minimum of 42 credits in courses numbered 300 and above",
        parent=ce_node,
        url=url,
        claim="The program requires at least 42 credits in courses numbered 300 and above.",
        critical=True,
    )

    # Capstone required
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="CE_Capstone_Required",
        desc="Requires a capstone design experience",
        parent=ce_node,
        url=url,
        claim="This program requires a capstone design experience (senior design).",
        critical=True,
    )

    # Capstone course codes
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="CE_Capstone_Course_ECIV_499R",
        desc="Capstone course code is ECIV 499R",
        parent=ce_node,
        url=url,
        claim="The specific capstone course code is ECIV 499R.",
        critical=True,
    )

    # Capstone total credits verified (uses answer's number if provided)
    capstone_total = (prog.capstone_total_credits or "").strip() if prog else ""
    capstone_claim = (
        f"The total number of credit hours for the capstone course ECIV 499R is {capstone_total} credit hours."
        if capstone_total else
        "The catalog page explicitly states the total number of credit hours for the capstone course ECIV 499R."
    )
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="CE_Capstone_Total_Credits_Verified",
        desc="States the total capstone credit hours and it matches the cited official MSU catalog source",
        parent=ce_node,
        url=url,
        claim=capstone_claim,
        critical=True,
        add_ins="If the answer did not provide a specific total, or the number does not match the catalog, mark Incorrect.",
    )

    # FE Exam required
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="CE_FE_Exam_Required",
        desc="Requires students to take the Fundamentals of Engineering (FE) Exam",
        parent=ce_node,
        url=url,
        claim="The program requires students to take the Fundamentals of Engineering (FE) exam.",
        critical=True,
    )

    # FE Exam EGEN 488 (0cr)
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="CE_FE_Exam_Registered_As_EGEN_488_0cr",
        desc="FE Exam is typically registered as EGEN 488 (0 credits)",
        parent=ce_node,
        url=url,
        claim="The FE exam is typically registered as EGEN 488 and carries 0 credits.",
        critical=True,
    )

    # FE Exam final semester
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="CE_FE_Exam_Final_Semester",
        desc="FE Exam is required during the final semester",
        parent=ce_node,
        url=url,
        claim="The FE exam must be taken during the program's final semester.",
        critical=True,
    )

    # Professional Electives minimum = 15
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="CE_Professional_Electives_Min_15",
        desc="Minimum professional elective credits required is 15",
        parent=ce_node,
        url=url,
        claim="The minimum number of professional elective credits required for this program is 15.",
        critical=True,
    )

    # At least 2 design-intensive courses
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="CE_Professional_Electives_Min_2_Design_Intensive_Courses",
        desc="Professional electives must include at least 2 design-intensive courses",
        parent=ce_node,
        url=url,
        claim="The program requires at least 2 design-intensive courses within the professional electives.",
        critical=True,
    )

    # Max 4 credits from Individual Problems/Internship/Research
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="CE_Professional_Electives_Max_4_Credits_IndivProblems_Internship_Research",
        desc="Professional electives allow a maximum of 4 credits from Individual Problems/Internships/Research",
        parent=ce_node,
        url=url,
        claim="At most 4 credits from Individual Problems, Internship, or Research may count toward professional electives.",
        critical=True,
    )

    # Four-year plan
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="CE_Four_Year_Semester_By_Semester_Plan",
        desc="Program has a documented semester-by-semester curriculum structure spanning four academic years",
        parent=ce_node,
        url=url,
        claim="The catalog page includes a semester-by-semester curriculum plan spanning four academic years.",
        critical=True,
    )

    # Source URL (simple)
    await _verify_leaf_simple(
        evaluator,
        leaf_id="CE_Source_URL",
        desc="Provides an official MSU academic catalog URL documenting these requirements",
        parent=ce_node,
        claim=f"The answer provides an official MSU academic catalog URL for the Civil Engineering program: {url}.",
        critical=True,
        add_ins="The URL should be from the official MSU catalog (e.g., catalog.montana.edu). If no URL is provided, mark Incorrect.",
    )


async def verify_chemical(evaluator: Evaluator, parent_node, prog: Optional[ProgramDetails]) -> None:
    cfg = EXPECTED_CONFIG["chemical"]
    url = prog.source_url if prog else None
    che_node = evaluator.add_parallel(
        id="Chemical_Engineering",
        desc="Chemical Engineering program requirements and documentation",
        parent=parent_node,
        critical=False,
    )

    # ChE_Name
    provided_name = prog.program_name if prog else ""
    await _verify_leaf_simple(
        evaluator,
        leaf_id="ChE_Name",
        desc="Program name is Chemical Engineering",
        parent=che_node,
        claim=f"The provided program name for Chemical is equivalent to 'Chemical Engineering'. Provided: '{provided_name}'.",
        critical=True,
        add_ins="Treat variations like 'Chemical Engineering BS' as equivalent. If no name is provided, mark Incorrect.",
    )

    # ABET EAC
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ChE_ABET_EAC",
        desc="Program is accredited by ABET Engineering Accreditation Commission (EAC)",
        parent=che_node,
        url=url,
        claim="This catalog page states the program is accredited by the Engineering Accreditation Commission (EAC) of ABET.",
        critical=True,
    )

    # College Affiliation
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ChE_College_Affiliation",
        desc="Program is housed in MSU's Norm Asbjornson College of Engineering",
        parent=che_node,
        url=url,
        claim="This catalog page indicates the program is housed within the Norm Asbjornson College of Engineering at Montana State University.",
        critical=True,
    )

    # Total Credits = 122
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ChE_Total_Credits_122",
        desc="Minimum total credits required for graduation is 122",
        parent=che_node,
        url=url,
        claim="The minimum total number of credits required for graduation in this program is 122.",
        critical=True,
    )

    # Upper Division 42
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ChE_Upper_Division_42",
        desc="Requires a minimum of 42 credits in courses numbered 300 and above",
        parent=che_node,
        url=url,
        claim="The program requires at least 42 credits in courses numbered 300 and above.",
        critical=True,
    )

    # Capstone required
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ChE_Capstone_Required",
        desc="Requires a capstone design experience",
        parent=che_node,
        url=url,
        claim="This program requires a capstone design experience (senior design).",
        critical=True,
    )

    # Capstone course codes
    expected_codes_text = _capstone_codes_text(cfg["capstone_codes"])
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ChE_Capstone_Courses_ECHM_411R_412R",
        desc="Capstone course code(s) are ECHM 411R and ECHM 412R",
        parent=che_node,
        url=url,
        claim=f"The specific capstone course codes are {expected_codes_text}.",
        critical=True,
    )

    # Capstone total credits verified (uses answer's number if provided)
    capstone_total = (prog.capstone_total_credits or "").strip() if prog else ""
    capstone_claim = (
        f"The total number of credit hours for the capstone sequence ({expected_codes_text}) is {capstone_total} in total."
        if capstone_total else
        "The catalog page explicitly states the total number of credit hours for the capstone sequence for this program."
    )
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ChE_Capstone_Total_Credits_Verified",
        desc="States the total capstone sequence credit hours and it matches the cited official MSU catalog source",
        parent=che_node,
        url=url,
        claim=capstone_claim,
        critical=True,
        add_ins="If the answer did not provide a specific total, or the number does not match the catalog, mark Incorrect.",
    )

    # FE Exam required
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ChE_FE_Exam_Required",
        desc="Requires students to take the Fundamentals of Engineering (FE) Exam",
        parent=che_node,
        url=url,
        claim="The program requires students to take the Fundamentals of Engineering (FE) exam.",
        critical=True,
    )

    # FE Exam EGEN 488 (0cr)
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ChE_FE_Exam_Registered_As_EGEN_488_0cr",
        desc="FE Exam is typically registered as EGEN 488 (0 credits)",
        parent=che_node,
        url=url,
        claim="The FE exam is typically registered as EGEN 488 and carries 0 credits.",
        critical=True,
    )

    # FE Exam final semester
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ChE_FE_Exam_Final_Semester",
        desc="FE Exam is required during the final semester",
        parent=che_node,
        url=url,
        claim="The FE exam must be taken during the program's final semester.",
        critical=True,
    )

    # Professional electives minimum credit count provided (not a fixed target; must be numeric and supported)
    pe_min = (prog.prof_electives_min_credits or "").strip() if prog else ""
    pe_claim = (
        f"The minimum number of professional elective credits required is {pe_min}."
        if pe_min else
        "The catalog page specifies a minimum number of professional elective credits for this program."
    )
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ChE_Professional_Electives_Specific_Credit_Count_Provided",
        desc="Professional elective requirements include a specific minimum credit count (a numeric value is stated)",
        parent=che_node,
        url=url,
        claim=pe_claim,
        critical=True,
        add_ins="If the answer does not provide a specific numeric minimum, mark Incorrect.",
    )

    # Additional constraints documented (non-critical)
    constraints_text = prog.prof_electives_constraints_text if prog else None
    constraints_claim = (
        f"The answer documents the additional constraints on professional electives for this program as: '{constraints_text}', "
        "and these constraints are supported by the catalog page. If the catalog has no additional constraints, stating 'none' is acceptable."
    )
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ChE_Professional_Electives_Additional_Constraints_Documented",
        desc="Any additional professional elective constraints are documented, or explicitly stated as none",
        parent=che_node,
        url=url,
        claim=constraints_claim,
        critical=False,
        add_ins="If the answer omits constraints while the catalog lists them, mark Incorrect.",
    )

    # Four-year plan
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="ChE_Four_Year_Semester_By_Semester_Plan",
        desc="Program has a documented semester-by-semester curriculum structure spanning four academic years",
        parent=che_node,
        url=url,
        claim="The catalog page includes a semester-by-semester curriculum plan spanning four academic years.",
        critical=True,
    )

    # Source URL
    await _verify_leaf_simple(
        evaluator,
        leaf_id="ChE_Source_URL",
        desc="Provides an official MSU academic catalog URL documenting these requirements",
        parent=che_node,
        claim=f"The answer provides an official MSU academic catalog URL for the Chemical Engineering program: {url}.",
        critical=True,
        add_ins="The URL should be from the official MSU catalog (e.g., catalog.montana.edu). If no URL is provided, mark Incorrect.",
    )


async def verify_electrical(evaluator: Evaluator, parent_node, prog: Optional[ProgramDetails]) -> None:
    cfg = EXPECTED_CONFIG["electrical"]
    url = prog.source_url if prog else None
    ee_node = evaluator.add_parallel(
        id="Electrical_Engineering",
        desc="Electrical Engineering program requirements and documentation",
        parent=parent_node,
        critical=False,
    )

    # EE_Name
    provided_name = prog.program_name if prog else ""
    await _verify_leaf_simple(
        evaluator,
        leaf_id="EE_Name",
        desc="Program name is Electrical Engineering",
        parent=ee_node,
        claim=f"The provided program name for Electrical is equivalent to 'Electrical Engineering'. Provided: '{provided_name}'.",
        critical=True,
        add_ins="Treat variations like 'Electrical Engineering BS' as equivalent. If no name is provided, mark Incorrect.",
    )

    # ABET EAC
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="EE_ABET_EAC",
        desc="Program is accredited by ABET Engineering Accreditation Commission (EAC)",
        parent=ee_node,
        url=url,
        claim="This catalog page states the program is accredited by the Engineering Accreditation Commission (EAC) of ABET.",
        critical=True,
    )

    # College Affiliation
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="EE_College_Affiliation",
        desc="Program is housed in MSU's Norm Asbjornson College of Engineering",
        parent=ee_node,
        url=url,
        claim="This catalog page indicates the program is housed within the Norm Asbjornson College of Engineering at Montana State University.",
        critical=True,
    )

    # Total Credits = 125
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="EE_Total_Credits_125",
        desc="Minimum total credits required for graduation is 125",
        parent=ee_node,
        url=url,
        claim="The minimum total number of credits required for graduation in this program is 125.",
        critical=True,
    )

    # Upper Division 42
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="EE_Upper_Division_42",
        desc="Requires a minimum of 42 credits in courses numbered 300 and above",
        parent=ee_node,
        url=url,
        claim="The program requires at least 42 credits in courses numbered 300 and above.",
        critical=True,
    )

    # Capstone required
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="EE_Capstone_Required",
        desc="Requires a capstone design experience",
        parent=ee_node,
        url=url,
        claim="This program requires a capstone design experience (senior design).",
        critical=True,
    )

    # Capstone course codes
    expected_codes_text = _capstone_codes_text(cfg["capstone_codes"])
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="EE_Capstone_Courses_EELE_488R_489R",
        desc="Capstone course code(s) are EELE 488R and EELE 489R",
        parent=ee_node,
        url=url,
        claim=f"The specific capstone course codes are {expected_codes_text}.",
        critical=True,
    )

    # Capstone total credits verified (uses answer's number if provided)
    capstone_total = (prog.capstone_total_credits or "").strip() if prog else ""
    capstone_claim = (
        f"The total number of credit hours for the capstone sequence ({expected_codes_text}) is {capstone_total} in total."
        if capstone_total else
        "The catalog page explicitly states the total number of credit hours for the capstone sequence for this program."
    )
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="EE_Capstone_Total_Credits_Verified",
        desc="States the total capstone sequence credit hours and it matches the cited official MSU catalog source",
        parent=ee_node,
        url=url,
        claim=capstone_claim,
        critical=True,
        add_ins="If the answer did not provide a specific total, or the number does not match the catalog, mark Incorrect.",
    )

    # FE Exam required
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="EE_FE_Exam_Required",
        desc="Requires students to take the Fundamentals of Engineering (FE) Exam",
        parent=ee_node,
        url=url,
        claim="The program requires students to take the Fundamentals of Engineering (FE) exam.",
        critical=True,
    )

    # FE Exam EGEN 488 (0cr)
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="EE_FE_Exam_Registered_As_EGEN_488_0cr",
        desc="FE Exam is typically registered as EGEN 488 (0 credits)",
        parent=ee_node,
        url=url,
        claim="The FE exam is typically registered as EGEN 488 and carries 0 credits.",
        critical=True,
    )

    # FE Exam final semester
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="EE_FE_Exam_Final_Semester",
        desc="FE Exam is required during the final semester",
        parent=ee_node,
        url=url,
        claim="The FE exam must be taken during the program's final semester.",
        critical=True,
    )

    # Professional Electives minimum = 27
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="EE_Professional_Electives_Min_27",
        desc="Minimum professional elective credits required is 27",
        parent=ee_node,
        url=url,
        claim="The minimum number of professional elective credits required for this program is 27.",
        critical=True,
    )

    # At least 18 credits in EE
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="EE_Professional_Electives_Min_18_EE_Credits",
        desc="Professional electives include at least 18 credits in EE",
        parent=ee_node,
        url=url,
        claim="Within the professional electives, at least 18 credits must be in Electrical Engineering (EE).",
        critical=True,
    )

    # At least 6 credits outside EE
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="EE_Professional_Electives_Min_6_Non_EE_Credits",
        desc="Professional electives include at least 6 credits outside EE",
        parent=ee_node,
        url=url,
        claim="Within the professional electives, at least 6 credits must be outside Electrical Engineering (non-EE).",
        critical=True,
    )

    # At least 11 credits at 300+ within PEs
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="EE_Professional_Electives_Min_11_Credits_300plus_within_PEs",
        desc="Professional electives package includes at least 11 credits at the 300-level or above",
        parent=ee_node,
        url=url,
        claim="Within the professional electives package, at least 11 credits must be at the 300-level or above.",
        critical=True,
    )

    # Four-year plan
    await _verify_leaf_by_url(
        evaluator,
        leaf_id="EE_Four_Year_Semester_By_Semester_Plan",
        desc="Program has a documented semester-by-semester curriculum structure spanning four academic years",
        parent=ee_node,
        url=url,
        claim="The catalog page includes a semester-by-semester curriculum plan spanning four academic years.",
        critical=True,
    )

    # Source URL
    await _verify_leaf_simple(
        evaluator,
        leaf_id="EE_Source_URL",
        desc="Provides an official MSU academic catalog URL documenting these requirements",
        parent=ee_node,
        claim=f"The answer provides an official MSU academic catalog URL for the Electrical Engineering program: {url}.",
        critical=True,
        add_ins="The URL should be from the official MSU catalog (e.g., catalog.montana.edu). If no URL is provided, mark Incorrect.",
    )


# --------------------------------------------------------------------------- #
# Distinctness check                                                          #
# --------------------------------------------------------------------------- #
def check_programs_distinct(extracted: FourProgramsExtraction) -> bool:
    names = [
        extracted.mechanical.program_name if extracted.mechanical else None,
        extracted.civil.program_name if extracted.civil else None,
        extracted.chemical.program_name if extracted.chemical else None,
        extracted.electrical.program_name if extracted.electrical else None,
    ]
    # All must be present and all distinct under normalization
    if any(n is None or str(n).strip() == "" for n in names):
        return False
    normed = [_norm_name(n) for n in names]
    return len(set(normed)) == 4


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
    Evaluate an answer for the MSU NACOE four programs task.
    """
    # Initialize evaluator
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

    # Extract structured info
    extracted: FourProgramsExtraction = await evaluator.extract(
        prompt=prompt_extract_four_programs(),
        template_class=FourProgramsExtraction,
        extraction_name="four_programs_extraction",
    )

    # Add the main task node
    task_node = evaluator.add_parallel(
        id="Four_Programs_Task",
        desc="Identify and document four distinct ABET-accredited undergraduate engineering programs at Montana State University per the given constraints",
        parent=root,
        critical=False,
    )

    # Programs_Are_Distinct (critical)
    evaluator.add_custom_node(
        result=check_programs_distinct(extracted),
        id="Programs_Are_Distinct",
        desc="All four identified programs are distinct from one another",
        parent=task_node,
        critical=True,
    )

    # Verify Mechanical
    await verify_mechanical(evaluator, task_node, extracted.mechanical)

    # Verify Civil
    await verify_civil(evaluator, task_node, extracted.civil)

    # Verify Chemical
    await verify_chemical(evaluator, task_node, extracted.chemical)

    # Verify Electrical
    await verify_electrical(evaluator, task_node, extracted.electrical)

    # Return summary
    return evaluator.get_summary()