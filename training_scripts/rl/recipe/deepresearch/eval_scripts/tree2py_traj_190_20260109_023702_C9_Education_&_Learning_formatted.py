import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "online_education_masters_programs"
TASK_DESCRIPTION = (
    "Identify 4 regionally accredited universities in the United States that offer fully online (100% asynchronous with no campus residency requirements) "
    "master's degree programs in Education or a related field (such as Educational Leadership, Educational Technology, Instructional Design, or School Counseling). "
    "Each program must meet ALL specified requirements and provide support URLs for each piece of information."
)

# Recognized regional accreditors list
RECOGNIZED_REGIONAL_ACCREDITORS = [
    "Higher Learning Commission", "HLC",
    "Middle States Commission on Higher Education", "MSCHE",
    "New England Commission of Higher Education", "NECHE",
    "Southern Association of Colleges and Schools Commission on Colleges", "SACSCOC",
    "WASC Senior College and University Commission", "WSCUC",
    "Northwest Commission on Colleges and Universities", "NWCCU",
    "Accrediting Commission for Community and Junior Colleges", "ACCJC"
]


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class Identification(BaseModel):
    university_name: Optional[str] = None
    program_title: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Accreditation(BaseModel):
    regional_accreditor: Optional[str] = None
    regional_urls: List[str] = Field(default_factory=list)
    is_counseling_program: Optional[bool] = None
    cacrep_status: Optional[str] = None  # e.g., "CACREP-accredited", "Not applicable"
    specialized_urls: List[str] = Field(default_factory=list)


class Delivery(BaseModel):
    online_asynchronous: Optional[str] = None  # confirmation string or "yes"
    no_residency: Optional[str] = None         # confirmation string or "yes"
    urls: List[str] = Field(default_factory=list)


class Structure(BaseModel):
    credit_hours: Optional[str] = None
    credit_hours_urls: List[str] = Field(default_factory=list)
    concentrations: List[str] = Field(default_factory=list)
    concentrations_urls: List[str] = Field(default_factory=list)
    completion_time: Optional[str] = None
    completion_time_urls: List[str] = Field(default_factory=list)
    start_dates: Optional[str] = None
    start_dates_urls: List[str] = Field(default_factory=list)


class Admissions(BaseModel):
    min_gpa: Optional[str] = None
    gpa_urls: List[str] = Field(default_factory=list)
    test_status: Optional[str] = None
    test_urls: List[str] = Field(default_factory=list)
    letters_required: Optional[str] = None
    letters_urls: List[str] = Field(default_factory=list)
    statement_required: Optional[str] = None
    statement_urls: List[str] = Field(default_factory=list)


class Faculty(BaseModel):
    qualifications: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Policies(BaseModel):
    transfer_credits: Optional[str] = None
    transfer_urls: List[str] = Field(default_factory=list)
    good_standing_gpa: Optional[str] = None
    good_standing_urls: List[str] = Field(default_factory=list)
    time_limit: Optional[str] = None
    time_limit_urls: List[str] = Field(default_factory=list)


class Financial(BaseModel):
    aid_type: Optional[str] = None
    aid_urls: List[str] = Field(default_factory=list)
    tuition_per_credit: Optional[str] = None
    tuition_urls: List[str] = Field(default_factory=list)


class Authorization(BaseModel):
    nc_sara: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Enrollment(BaseModel):
    fulltime_hours: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Graduation(BaseModel):
    requirement_type: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ProgramInfo(BaseModel):
    identification: Identification = Field(default_factory=Identification)
    accreditation: Accreditation = Field(default_factory=Accreditation)
    delivery: Delivery = Field(default_factory=Delivery)
    structure: Structure = Field(default_factory=Structure)
    admissions: Admissions = Field(default_factory=Admissions)
    faculty: Faculty = Field(default_factory=Faculty)
    policies: Policies = Field(default_factory=Policies)
    financial: Financial = Field(default_factory=Financial)
    authorization: Authorization = Field(default_factory=Authorization)
    enrollment: Enrollment = Field(default_factory=Enrollment)
    graduation: Graduation = Field(default_factory=Graduation)


class ProgramsExtraction(BaseModel):
    programs: List[ProgramInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return (
        "Extract up to 4 qualifying master's programs in Education or closely related fields from the answer. "
        "For each program, return the following JSON object fields as strings and URL lists, exactly as appear in the answer. "
        "If a field is not provided, use null for the value and an empty list for URL arrays.\n\n"
        "Structure to return:\n"
        "{\n"
        "  \"programs\": [\n"
        "    {\n"
        "      \"identification\": {\n"
        "        \"university_name\": string|null,\n"
        "        \"program_title\": string|null,\n"
        "        \"urls\": [url,...]\n"
        "      },\n"
        "      \"accreditation\": {\n"
        "        \"regional_accreditor\": string|null,\n"
        "        \"regional_urls\": [url,...],\n"
        "        \"is_counseling_program\": boolean|null,\n"
        "        \"cacrep_status\": string|null,\n"
        "        \"specialized_urls\": [url,...]\n"
        "      },\n"
        "      \"delivery\": {\n"
        "        \"online_asynchronous\": string|null,\n"
        "        \"no_residency\": string|null,\n"
        "        \"urls\": [url,...]\n"
        "      },\n"
        "      \"structure\": {\n"
        "        \"credit_hours\": string|null,\n"
        "        \"credit_hours_urls\": [url,...],\n"
        "        \"concentrations\": [string,...],\n"
        "        \"concentrations_urls\": [url,...],\n"
        "        \"completion_time\": string|null,\n"
        "        \"completion_time_urls\": [url,...],\n"
        "        \"start_dates\": string|null,\n"
        "        \"start_dates_urls\": [url,...]\n"
        "      },\n"
        "      \"admissions\": {\n"
        "        \"min_gpa\": string|null,\n"
        "        \"gpa_urls\": [url,...],\n"
        "        \"test_status\": string|null,\n"
        "        \"test_urls\": [url,...],\n"
        "        \"letters_required\": string|null,\n"
        "        \"letters_urls\": [url,...],\n"
        "        \"statement_required\": string|null,\n"
        "        \"statement_urls\": [url,...]\n"
        "      },\n"
        "      \"faculty\": {\n"
        "        \"qualifications\": string|null,\n"
        "        \"urls\": [url,...]\n"
        "      },\n"
        "      \"policies\": {\n"
        "        \"transfer_credits\": string|null,\n"
        "        \"transfer_urls\": [url,...],\n"
        "        \"good_standing_gpa\": string|null,\n"
        "        \"good_standing_urls\": [url,...],\n"
        "        \"time_limit\": string|null,\n"
        "        \"time_limit_urls\": [url,...]\n"
        "      },\n"
        "      \"financial\": {\n"
        "        \"aid_type\": string|null,\n"
        "        \"aid_urls\": [url,...],\n"
        "        \"tuition_per_credit\": string|null,\n"
        "        \"tuition_urls\": [url,...]\n"
        "      },\n"
        "      \"authorization\": {\n"
        "        \"nc_sara\": string|null,\n"
        "        \"urls\": [url,...]\n"
        "      },\n"
        "      \"enrollment\": {\n"
        "        \"fulltime_hours\": string|null,\n"
        "        \"urls\": [url,...]\n"
        "      },\n"
        "      \"graduation\": {\n"
        "        \"requirement_type\": string|null,\n"
        "        \"urls\": [url,...]\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Notes:\n"
        "- Only include URLs explicitly present in the answer (markdown links are okay). If no URL is present for an item, leave the corresponding URL list empty.\n"
        "- Set is_counseling_program to true if the program is School Counseling or Clinical Mental Health Counseling; otherwise false.\n"
        "- For test_status, use statements like 'GRE not required' or 'test-optional' if provided.\n"
        "- Use plain strings for numbers (e.g., credit hours '30', GPA '3.0', tuition '$650'), do not convert to numbers.\n"
        "- If more than 4 programs are present, include only the first 4."
    )


# --------------------------------------------------------------------------- #
# Helper parsing functions                                                    #
# --------------------------------------------------------------------------- #
def _has_text(x: Optional[str]) -> bool:
    return bool(x and str(x).strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def _extract_first_number(text: Optional[str]) -> Optional[float]:
    if not _has_text(text):
        return None
    m = re.findall(r"(\d+(?:\.\d+)?)", text)
    if not m:
        # try words for small numbers (two/three)
        word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
        for w, v in word_map.items():
            if w in text.lower():
                return float(v)
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def _parse_credit_hours(text: Optional[str]) -> Optional[int]:
    num = _extract_first_number(text)
    return int(num) if num is not None else None


def _parse_tuition_per_credit(text: Optional[str]) -> Optional[float]:
    if not _has_text(text):
        return None
    # remove $ and commas
    cleaned = re.sub(r"[\$,]", "", text)
    num = _extract_first_number(cleaned)
    return float(num) if num is not None else None


def _parse_letters_required(text: Optional[str]) -> Optional[int]:
    num = _extract_first_number(text)
    return int(num) if num is not None else None


def _parse_gpa(text: Optional[str]) -> Optional[float]:
    num = _extract_first_number(text)
    return float(num) if num is not None else None


def _parse_time_limit_years(text: Optional[str]) -> Optional[float]:
    if not _has_text(text):
        return None
    # handle years or months
    m_num = _extract_first_number(text)
    if m_num is None:
        return None
    t = text.lower()
    if "month" in t:
        # convert months to years
        return float(m_num) / 12.0
    # assume years if says years or default
    return float(m_num)


def _parse_completion_months(text: Optional[str]) -> Optional[float]:
    if not _has_text(text):
        return None
    num = _extract_first_number(text)
    if num is None:
        return None
    t = text.lower()
    if "year" in t:
        return float(num) * 12.0
    # default to months
    return float(num)


def _parse_fulltime_hours(text: Optional[str]) -> Optional[int]:
    return _parse_credit_hours(text)


def _is_valid_regional_accreditor(name: Optional[str]) -> bool:
    if not _has_text(name):
        return False
    lower = name.strip().lower()
    for acc in RECOGNIZED_REGIONAL_ACCREDITORS:
        if lower == acc.lower() or acc.lower() in lower:
            return True
    return False


# --------------------------------------------------------------------------- #
# Small helper to add URL existence nodes                                     #
# --------------------------------------------------------------------------- #
def add_url_presence_node(
    evaluator: Evaluator,
    node_id: str,
    desc: str,
    urls: List[str],
    parent_node,
    critical: bool = True
):
    evaluator.add_custom_node(
        result=_has_urls(urls),
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical
    )


# --------------------------------------------------------------------------- #
# Verification builder for one program                                        #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    root_parent,
    program: ProgramInfo,
    index: int
) -> None:
    pid = index + 1
    # Program main node (allow partial at root; inside requirements are marked critical)
    program_node = evaluator.add_parallel(
        id=f"program_{pid}",
        desc=f"Program #{pid} verification",
        parent=root_parent,
        critical=False
    )

    # ---------------- Identification ----------------
    ident_node = evaluator.add_parallel(
        id=f"program_{pid}_identification",
        desc="University name and specific program title are provided",
        parent=program_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(program.identification.university_name),
        id=f"program_{pid}_university_name",
        desc="University name is provided",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(program.identification.program_title),
        id=f"program_{pid}_program_title",
        desc="Specific program title is provided",
        parent=ident_node,
        critical=True
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_identification_url",
        "URL reference supporting university and program identification",
        program.identification.urls, ident_node, critical=True
    )

    # ---------------- Accreditation ----------------
    accred_node = evaluator.add_parallel(
        id=f"program_{pid}_accreditation",
        desc="Accreditation requirements are met",
        parent=program_node,
        critical=True
    )
    # Regional accreditation
    regional_node = evaluator.add_parallel(
        id=f"program_{pid}_regional_accreditation",
        desc="University is regionally accredited by one of the seven CHEA-recognized regional accreditors",
        parent=accred_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_valid_regional_accreditor(program.accreditation.regional_accreditor),
        id=f"program_{pid}_regional_accreditor_identified",
        desc="The specific regional accreditor is identified",
        parent=regional_node,
        critical=True
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_regional_accreditation_url",
        "URL reference confirming regional accreditation",
        program.accreditation.regional_urls, regional_node, critical=True
    )
    # Specialized accreditation (CACREP)
    spec_node = evaluator.add_parallel(
        id=f"program_{pid}_specialized_accreditation",
        desc="If program is in counseling, CACREP accreditation is confirmed; otherwise this requirement is not applicable",
        parent=accred_node,
        critical=True
    )
    # Status addressed: pass if not counseling OR (counseling AND CACREP string present)
    cacrep_addressed = (program.accreditation.is_counseling_program is False) or (
        program.accreditation.is_counseling_program is True and _has_text(program.accreditation.cacrep_status)
    )
    evaluator.add_custom_node(
        result=cacrep_addressed,
        id=f"program_{pid}_cacrep_status",
        desc="CACREP accreditation status is addressed (confirmed if counseling program, or stated as not applicable)",
        parent=spec_node,
        critical=True
    )
    # URL presence only if counseling; if not counseling, not required
    spec_urls_ok = (program.accreditation.is_counseling_program is False) or _has_urls(program.accreditation.specialized_urls)
    evaluator.add_custom_node(
        result=spec_urls_ok,
        id=f"program_{pid}_specialized_accreditation_url",
        desc="URL reference for specialized accreditation status",
        parent=spec_node,
        critical=True
    )

    # ---------------- Delivery format ----------------
    delivery_node = evaluator.add_parallel(
        id=f"program_{pid}_delivery_format",
        desc="Program delivery format meets requirements",
        parent=program_node,
        critical=True
    )
    leaf_online = evaluator.add_leaf(
        id=f"program_{pid}_online_asynchronous",
        desc="Program is confirmed as 100% online and asynchronous",
        parent=delivery_node,
        critical=True
    )
    online_claim = f"The program '{program.identification.program_title or ''}' is offered 100% online in an asynchronous format (no required live sessions)."
    await evaluator.verify(
        claim=online_claim,
        node=leaf_online,
        sources=program.delivery.urls,
        additional_instruction="Pass if the page explicitly or clearly implies fully online and asynchronous delivery. Accept synonyms like 'fully online', 'asynchronous', 'no scheduled live sessions'."
    )
    leaf_no_res = evaluator.add_leaf(
        id=f"program_{pid}_no_residency",
        desc="Program requires no campus visits or residency",
        parent=delivery_node,
        critical=True
    )
    no_res_claim = f"The program '{program.identification.program_title or ''}' requires no campus visits or residency."
    await evaluator.verify(
        claim=no_res_claim,
        node=leaf_no_res,
        sources=program.delivery.urls,
        additional_instruction="Pass if the page states there are no campus visits, residencies, or in-person requirements for the program."
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_delivery_format_url",
        "URL reference confirming delivery format",
        program.delivery.urls, delivery_node, critical=True
    )

    # ---------------- Structure ----------------
    struct_node = evaluator.add_parallel(
        id=f"program_{pid}_structure",
        desc="Program structure requirements are met",
        parent=program_node,
        critical=True
    )
    # Credit hours
    credit_node = evaluator.add_parallel(
        id=f"program_{pid}_credit_hours",
        desc="Minimum credit hours requirement (at least 30) is stated and met",
        parent=struct_node,
        critical=True
    )
    ch_val = _parse_credit_hours(program.structure.credit_hours)
    evaluator.add_custom_node(
        result=(ch_val is not None and ch_val >= 30),
        id=f"program_{pid}_credit_hours_value",
        desc="The specific minimum credit hours required is stated",
        parent=credit_node,
        critical=True
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_credit_hours_url",
        "URL reference for credit hours requirement",
        program.structure.credit_hours_urls, credit_node, critical=True
    )
    # Concentrations
    conc_node = evaluator.add_parallel(
        id=f"program_{pid}_concentrations",
        desc="At least 2 concentrations/specializations are offered",
        parent=struct_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(len(program.structure.concentrations) >= 2 and _has_text(program.structure.concentrations[0]) and _has_text(program.structure.concentrations[1])),
        id=f"program_{pid}_concentration_examples",
        desc="Two specific concentration/specialization names are provided",
        parent=conc_node,
        critical=True
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_concentrations_url",
        "URL reference for concentrations/specializations offered",
        program.structure.concentrations_urls, conc_node, critical=True
    )
    # Completion time
    comp_node = evaluator.add_parallel(
        id=f"program_{pid}_completion_time",
        desc="Program can be completed within 24 months full-time",
        parent=struct_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(program.structure.completion_time),
        id=f"program_{pid}_completion_time_stated",
        desc="Expected completion time for full-time students is stated",
        parent=comp_node,
        critical=True
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_completion_time_url",
        "URL reference for program completion time",
        program.structure.completion_time_urls, comp_node, critical=True
    )
    leaf_completion = evaluator.add_leaf(
        id=f"program_{pid}_completion_time_check",
        desc="Completion time within 24 months verified",
        parent=comp_node,
        critical=True
    )
    completion_claim = "The program can be completed within 24 months or less when studying full-time."
    await evaluator.verify(
        claim=completion_claim,
        node=leaf_completion,
        sources=program.structure.completion_time_urls,
        additional_instruction="Pass if the page shows completion in 24 months or less (e.g., '18 months', '2 years')."
    )
    # Start dates
    start_node = evaluator.add_parallel(
        id=f"program_{pid}_start_dates",
        desc="Program offers both fall and spring start dates",
        parent=struct_node,
        critical=True
    )
    leaf_start = evaluator.add_leaf(
        id=f"program_{pid}_start_dates_confirmed",
        desc="Both fall and spring start dates are confirmed",
        parent=start_node,
        critical=True
    )
    start_claim = "The program offers start dates in both the fall and spring semesters."
    await evaluator.verify(
        claim=start_claim,
        node=leaf_start,
        sources=program.structure.start_dates_urls,
        additional_instruction="Pass if the page indicates start terms include both Fall and Spring."
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_start_dates_url",
        "URL reference for start dates",
        program.structure.start_dates_urls, start_node, critical=True
    )

    # ---------------- Admissions ----------------
    adm_node = evaluator.add_parallel(
        id=f"program_{pid}_admission",
        desc="Admission requirements meet specifications",
        parent=program_node,
        critical=True
    )
    # GPA requirement
    gpa_node = evaluator.add_parallel(
        id=f"program_{pid}_gpa_requirement",
        desc="Minimum GPA requirement is 3.0 or lower",
        parent=adm_node,
        critical=True
    )
    gpa_val = _parse_gpa(program.admissions.min_gpa)
    evaluator.add_custom_node(
        result=(gpa_val is not None and gpa_val <= 3.0),
        id=f"program_{pid}_gpa_value",
        desc="The specific minimum GPA requirement is stated",
        parent=gpa_node,
        critical=True
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_gpa_url",
        "URL reference for GPA requirement",
        program.admissions.gpa_urls, gpa_node, critical=True
    )
    # Test optional / no GRE/GMAT
    test_node = evaluator.add_parallel(
        id=f"program_{pid}_test_optional",
        desc="GRE/GMAT is not required for admission",
        parent=adm_node,
        critical=True
    )
    leaf_test = evaluator.add_leaf(
        id=f"program_{pid}_test_status",
        desc="Test-optional or no-test-required status is confirmed",
        parent=test_node,
        critical=True
    )
    test_claim = "GRE or GMAT are not required for admission to this program (test-optional or no standardized tests required)."
    await evaluator.verify(
        claim=test_claim,
        node=leaf_test,
        sources=program.admissions.test_urls,
        additional_instruction="Pass if the page explicitly states GRE/GMAT not required or the program is test-optional/no-test-required."
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_test_url",
        "URL reference for test requirements",
        program.admissions.test_urls, test_node, critical=True
    )
    # Recommendations
    rec_node = evaluator.add_parallel(
        id=f"program_{pid}_recommendations",
        desc="2-3 letters of recommendation are required",
        parent=adm_node,
        critical=True
    )
    letters_num = _parse_letters_required(program.admissions.letters_required)
    evaluator.add_custom_node(
        result=(letters_num is not None and 2 <= letters_num <= 3),
        id=f"program_{pid}_recommendations_number",
        desc="The specific number of letters of recommendation required is stated",
        parent=rec_node,
        critical=True
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_recommendations_url",
        "URL reference for recommendation requirements",
        program.admissions.letters_urls, rec_node, critical=True
    )
    # Statement of purpose
    stmt_node = evaluator.add_parallel(
        id=f"program_{pid}_statement",
        desc="Personal statement or statement of purpose is required",
        parent=adm_node,
        critical=True
    )
    leaf_stmt = evaluator.add_leaf(
        id=f"program_{pid}_statement_confirmed",
        desc="Statement requirement is confirmed",
        parent=stmt_node,
        critical=True
    )
    stmt_claim = "A personal statement or statement of purpose is required for admission to this program."
    await evaluator.verify(
        claim=stmt_claim,
        node=leaf_stmt,
        sources=program.admissions.statement_urls,
        additional_instruction="Pass if the page lists a personal statement/statement of purpose as a required admission material."
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_statement_url",
        "URL reference for statement requirement",
        program.admissions.statement_urls, stmt_node, critical=True
    )

    # ---------------- Faculty ----------------
    faculty_node = evaluator.add_parallel(
        id=f"program_{pid}_faculty",
        desc="Faculty hold terminal degrees",
        parent=program_node,
        critical=True
    )
    leaf_fac = evaluator.add_leaf(
        id=f"program_{pid}_faculty_qualifications",
        desc="Faculty are confirmed to hold terminal degrees in education or related fields",
        parent=faculty_node,
        critical=True
    )
    fac_claim = "Program faculty hold terminal degrees (PhD, EdD, or equivalent) in education or related fields."
    await evaluator.verify(
        claim=fac_claim,
        node=leaf_fac,
        sources=program.faculty.urls,
        additional_instruction="Pass if faculty qualifications indicate terminal degrees such as PhD/EdD or equivalent."
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_faculty_url",
        "URL reference for faculty qualifications",
        program.faculty.urls, faculty_node, critical=True
    )

    # ---------------- Policies ----------------
    policies_node = evaluator.add_parallel(
        id=f"program_{pid}_policies",
        desc="Academic policies meet requirements",
        parent=program_node,
        critical=True
    )
    # Transfer credits
    transfer_node = evaluator.add_parallel(
        id=f"program_{pid}_transfer_credits",
        desc="At least 6 credit hours can be transferred from other institutions",
        parent=policies_node,
        critical=True
    )
    transfer_val = _parse_credit_hours(program.policies.transfer_credits)
    evaluator.add_custom_node(
        result=(transfer_val is not None and transfer_val >= 6),
        id=f"program_{pid}_transfer_amount",
        desc="The number of transferable credits is stated",
        parent=transfer_node,
        critical=True
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_transfer_url",
        "URL reference for transfer credit policy",
        program.policies.transfer_urls, transfer_node, critical=True
    )
    # Minimum GPA for good standing
    standing_node = evaluator.add_parallel(
        id=f"program_{pid}_minimum_gpa",
        desc="Minimum 3.0 GPA must be maintained for good standing",
        parent=policies_node,
        critical=True
    )
    leaf_standing = evaluator.add_leaf(
        id=f"program_{pid}_good_standing_gpa",
        desc="The minimum GPA for good standing is stated",
        parent=standing_node,
        critical=True
    )
    standing_claim = "Students must maintain a minimum GPA of 3.0 to remain in good standing in the program."
    await evaluator.verify(
        claim=standing_claim,
        node=leaf_standing,
        sources=program.policies.good_standing_urls,
        additional_instruction="Pass if the policy page states 3.0 minimum GPA for good academic standing."
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_good_standing_url",
        "URL reference for academic standing requirements",
        program.policies.good_standing_urls, standing_node, critical=True
    )
    # Time limit
    time_node = evaluator.add_parallel(
        id=f"program_{pid}_time_limit",
        desc="Maximum completion time is at least 5 years",
        parent=policies_node,
        critical=True
    )
    years_val = _parse_time_limit_years(program.policies.time_limit)
    evaluator.add_custom_node(
        result=(years_val is not None and years_val >= 5.0),
        id=f"program_{pid}_time_limit_stated",
        desc="The maximum time to complete is stated",
        parent=time_node,
        critical=True
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_time_limit_url",
        "URL reference for time limit policy",
        program.policies.time_limit_urls, time_node, critical=True
    )

    # ---------------- Financial ----------------
    financial_node = evaluator.add_parallel(
        id=f"program_{pid}_financial",
        desc="Financial aid and tuition requirements are met",
        parent=program_node,
        critical=True
    )
    # Financial aid
    aid_node = evaluator.add_parallel(
        id=f"program_{pid}_financial_aid",
        desc="At least one form of graduate financial assistance is available",
        parent=financial_node,
        critical=True
    )
    leaf_aid = evaluator.add_leaf(
        id=f"program_{pid}_aid_type",
        desc="Type of financial assistance available is stated",
        parent=aid_node,
        critical=True
    )
    aid_claim = "At least one form of graduate financial assistance (assistantships, fellowships, scholarships, or institutional grants) is available to online students."
    await evaluator.verify(
        claim=aid_claim,
        node=leaf_aid,
        sources=program.financial.aid_urls,
        additional_instruction="Pass if any listed financial aid option applies to graduate online students."
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_aid_url",
        "URL reference for financial aid options",
        program.financial.aid_urls, aid_node, critical=True
    )
    # Tuition per credit <= $1000
    tuition_node = evaluator.add_parallel(
        id=f"program_{pid}_tuition",
        desc="Per-credit-hour tuition is $1,000 or less",
        parent=financial_node,
        critical=True
    )
    tuition_val = _parse_tuition_per_credit(program.financial.tuition_per_credit)
    evaluator.add_custom_node(
        result=(tuition_val is not None and tuition_val <= 1000.0),
        id=f"program_{pid}_tuition_rate",
        desc="The specific per-credit-hour tuition rate is stated",
        parent=tuition_node,
        critical=True
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_tuition_url",
        "URL reference for tuition costs",
        program.financial.tuition_urls, tuition_node, critical=True
    )

    # ---------------- Authorization (NC-SARA) ----------------
    auth_node = evaluator.add_parallel(
        id=f"program_{pid}_authorization",
        desc="State authorization requirement is met",
        parent=program_node,
        critical=True
    )
    leaf_sara = evaluator.add_leaf(
        id=f"program_{pid}_nc_sara",
        desc="University participates in NC-SARA",
        parent=auth_node,
        critical=True
    )
    sara_claim = "The university participates in NC-SARA (National Council for State Authorization Reciprocity Agreements) for distance education."
    await evaluator.verify(
        claim=sara_claim,
        node=leaf_sara,
        sources=program.authorization.urls,
        additional_instruction="Pass if the page confirms NC-SARA participation (the university appears on NC-SARA listings or states membership)."
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_authorization_url",
        "URL reference for NC-SARA participation",
        program.authorization.urls, auth_node, critical=True
    )

    # ---------------- Enrollment ----------------
    enroll_node = evaluator.add_parallel(
        id=f"program_{pid}_enrollment",
        desc="Full-time enrollment definition meets requirement",
        parent=program_node,
        critical=True
    )
    leaf_fulltime = evaluator.add_leaf(
        id=f"program_{pid}_fulltime_hours",
        desc="Full-time graduate status is defined as 9 or fewer credit hours per semester",
        parent=enroll_node,
        critical=True
    )
    fulltime_claim = "Full-time graduate enrollment is defined as 9 or fewer credit hours per semester."
    await evaluator.verify(
        claim=fulltime_claim,
        node=leaf_fulltime,
        sources=program.enrollment.urls,
        additional_instruction="Pass if the page defines full-time as 9 credits per term (or fewer). Accept common phrasing like '9 credits is full-time'."
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_enrollment_url",
        "URL reference for enrollment status definitions",
        program.enrollment.urls, enroll_node, critical=True
    )

    # ---------------- Graduation ----------------
    grad_node = evaluator.add_parallel(
        id=f"program_{pid}_graduation",
        desc="Graduation requirements meet specifications",
        parent=program_node,
        critical=True
    )
    req_node = evaluator.add_parallel(
        id=f"program_{pid}_culminating_requirement",
        desc="Non-thesis option with capstone, comprehensive exam, or portfolio is offered",
        parent=grad_node,
        critical=True
    )
    leaf_req_type = evaluator.add_leaf(
        id=f"program_{pid}_requirement_type",
        desc="The specific culminating requirement type is stated",
        parent=req_node,
        critical=True
    )
    req_claim = f"The program offers a non-thesis option with a culminating requirement such as a {program.graduation.requirement_type or 'capstone/comprehensive exam/portfolio'}."
    await evaluator.verify(
        claim=req_claim,
        node=leaf_req_type,
        sources=program.graduation.urls,
        additional_instruction="Pass if the page mentions non-thesis culminating options (capstone, comprehensive exam, portfolio)."
    )
    add_url_presence_node(
        evaluator, f"program_{pid}_graduation_url",
        "URL reference for graduation requirements",
        program.graduation.urls, req_node, critical=True
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
    # Initialize evaluator (root set to PARALLEL and non-critical to allow partial scoring across programs)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Evaluate whether 4 qualifying online master's degree programs in Education have been identified with complete information",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction"
    )

    # Record ground-truth support info (recognized accreditors)
    evaluator.add_ground_truth({
        "recognized_regional_accreditors": RECOGNIZED_REGIONAL_ACCREDITORS,
        "requirements_summary": {
            "credit_hours_min": 30,
            "completion_months_max": 24,
            "good_standing_gpa_min": 3.0,
            "transfer_credits_min": 6,
            "tuition_per_credit_max_usd": 1000.0,
            "fulltime_hours_max": 9,
            "time_limit_years_min": 5
        }
    }, gt_type="reference_requirements")

    # Prepare up to 4 programs (pad if fewer, truncate if more)
    programs: List[ProgramInfo] = extracted.programs[:4]
    while len(programs) < 4:
        programs.append(ProgramInfo())

    # Build verification tree for each program
    for i, prog in enumerate(programs):
        await verify_program(evaluator, root, prog, i)

    # Return summary
    return evaluator.get_summary()