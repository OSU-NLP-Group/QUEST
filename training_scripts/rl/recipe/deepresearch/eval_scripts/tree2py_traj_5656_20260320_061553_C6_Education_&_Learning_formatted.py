import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nc_promise_principal_pathway"
TASK_DESCRIPTION = (
    "As a North Carolina resident planning a career as a K-12 school principal, you want to minimize undergraduate costs "
    "while meeting all requirements for principal licensure. Identify a complete educational pathway that satisfies the criteria, "
    "including NC Promise undergraduate institution and tuition details (2025-26 or 2026-27), post-licensure K-12 teaching "
    "experience (≥3 years), and a master's program in educational leadership/administration that leads to principal licensure. "
    "Provide program details, GPA requirement, credit hours, and official reference URLs for undergraduate, teaching requirements, "
    "graduate program, and licensure requirements."
)

ALLOWED_NC_PROMISE_SCHOOLS = {
    "elizabeth city state university",
    "fayetteville state university",
    "university of north carolina at pembroke",
    "unc pembroke",
    "western carolina university",
}

ELIGIBLE_AY_LABELS = {"2025-2026", "2026-2027", "ay 2025-2026", "ay 2026-2027", "2025/2026", "2026/2027", "2025–2026", "2026–2027", "2025-26", "2026-27"}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ReferenceURLs(BaseModel):
    undergrad_url: Optional[str] = None
    teaching_url: Optional[str] = None
    graduate_url: Optional[str] = None
    licensure_url: Optional[str] = None


class UndergraduateInfo(BaseModel):
    university_name: Optional[str] = None
    program_name: Optional[str] = None
    program_leads_to_teaching_license: Optional[str] = None  # yes/no/unclear
    is_regionally_accredited: Optional[str] = None  # yes/no/unclear
    nc_promise_tuition_per_semester: Optional[str] = None  # e.g., "$500"
    tuition_academic_year: Optional[str] = None  # e.g., "2025-2026" or "2026-2027"
    eight_semester_total: Optional[str] = None  # e.g., "$4,000"


class TeachingInfo(BaseModel):
    obtained_teaching_license_after_bachelors: Optional[str] = None  # yes/no/unclear
    post_licensure_experience_years: Optional[str] = None  # e.g., "3"


class GraduateInfo(BaseModel):
    university_name: Optional[str] = None
    program_name_and_type: Optional[str] = None  # e.g., "M.Ed. in Educational Leadership"
    is_masters_level: Optional[str] = None  # yes/no/unclear
    field: Optional[str] = None  # e.g., "Educational Leadership"
    leads_to_principal_licensure: Optional[str] = None  # yes/no/unclear
    admission_min_gpa: Optional[str] = None  # e.g., "3.0"
    total_credit_hours: Optional[str] = None  # e.g., "30"
    is_regionally_accredited: Optional[str] = None  # yes/no/unclear


class PathwayExtraction(BaseModel):
    undergraduate: Optional[UndergraduateInfo] = None
    teaching: Optional[TeachingInfo] = None
    graduate: Optional[GraduateInfo] = None
    references: Optional[ReferenceURLs] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_pathway() -> str:
    return """
    Extract the complete educational pathway data elements exactly as stated in the answer. Return a JSON object with the following structure:

    {
      "undergraduate": {
        "university_name": string or null,
        "program_name": string or null,
        "program_leads_to_teaching_license": "yes" | "no" | "unclear" | null,
        "is_regionally_accredited": "yes" | "no" | "unclear" | null,
        "nc_promise_tuition_per_semester": string or null,  // e.g., "$500" or "500 USD"
        "tuition_academic_year": string or null,            // must be one of 2025-2026 or 2026-2027 (allow variants like 2025–2026, AY 2025/2026)
        "eight_semester_total": string or null              // e.g., "$4,000", the total tuition for 8 semesters based on NC Promise per-semester rate
      },
      "teaching": {
        "obtained_teaching_license_after_bachelors": "yes" | "no" | "unclear" | null,
        "post_licensure_experience_years": string or null   // numeric years as a string, e.g., "3"
      },
      "graduate": {
        "university_name": string or null,
        "program_name_and_type": string or null,            // include both program name and degree type (e.g., "M.Ed. in Educational Leadership")
        "is_masters_level": "yes" | "no" | "unclear" | null,
        "field": string or null,                            // e.g., "Educational Leadership" or "Educational Administration"
        "leads_to_principal_licensure": "yes" | "no" | "unclear" | null,
        "admission_min_gpa": string or null,                // numeric as string, e.g., "3.0"
        "total_credit_hours": string or null,               // numeric as string, e.g., "30" or "36"
        "is_regionally_accredited": "yes" | "no" | "unclear" | null
      },
      "references": {
        "undergrad_url": string or null,   // official source for undergrad/program/tuition
        "teaching_url": string or null,    // official source for teaching license/teaching requirements
        "graduate_url": string or null,    // official source for graduate program
        "licensure_url": string or null    // official state/agency page for principal/administrator licensure requirements
      }
    }

    Rules:
    - Extract only what is explicitly stated in the answer.
    - If a value is missing, set it to null.
    - Preserve currency symbols and year formats as written.
    - For boolean-like fields, use "yes"/"no"/"unclear".
    - For years, if the answer references the 2025–2026 or 2026–2027 academic year, record it verbatim (allow forms like "2025-2026", "2025/2026", "AY 2025–2026").
    - Extract exactly one URL for each of the four 'references' fields if provided; otherwise null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _string_truthy(s: Optional[str]) -> bool:
    if not s:
        return False
    return s.strip().lower() in {"yes", "true", "y", "1"}


def _parse_number(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    ss = s.strip().lower()
    # Keep digits, dot, and comma; remove non-numeric symbols except decimal separators
    # Replace commas used as thousands separators
    ss = ss.replace(",", "")
    m = re.findall(r"[-+]?\d*\.?\d+", ss)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def _normalize_school_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return name.strip().lower()


def _is_nc_promise_school(name: Optional[str]) -> bool:
    n = _normalize_school_name(name)
    if not n:
        return False
    # handle common synonyms
    synonyms = {
        "unc pembroke": "university of north carolina at pembroke",
        "the university of north carolina at pembroke": "university of north carolina at pembroke",
        "wcu": "western carolina university",
        "fsu": "fayetteville state university",
        "ecsu": "elizabeth city state university",
    }
    n = synonyms.get(n, n)
    return n in ALLOWED_NC_PROMISE_SCHOOLS


def _year_is_eligible(year_str: Optional[str]) -> bool:
    if not year_str:
        return False
    s = year_str.strip().lower().replace("–", "-").replace("—", "-").replace(" ", "")
    # Normalize slashes to dashes for comparison
    s = s.replace("/", "-")
    # Accept patterns containing both years
    if "2025" in s and "2026" in s:
        return True
    if "2026" in s and "2027" in s:
        return True
    # Accept common shorthand 2025-26, 2026-27
    if s in {"2025-26", "ay2025-26", "2026-27", "ay2026-27"}:
        return True
    # Fallback membership check
    return s in {x.replace(" ", "").replace("/", "-").lower() for x in ELIGIBLE_AY_LABELS}


def _year_is_2025_26(year_str: Optional[str]) -> bool:
    if not year_str:
        return False
    s = year_str.strip().lower().replace("–", "-").replace(" ", "").replace("/", "-")
    return ("2025" in s and "2026" in s) or (s in {"2025-26", "ay2025-26"})


def _credits_in_typical_range(credits: Optional[str]) -> Optional[bool]:
    v = _parse_number(credits)
    if v is None:
        return None
    return 30.0 <= v <= 36.0


def _gpa_at_least_min(gpa_str: Optional[str], min_required: float = 3.0) -> Optional[bool]:
    v = _parse_number(gpa_str)
    if v is None:
        return None
    return v >= min_required


def _calc_total_ok(per_sem_str: Optional[str], total_str: Optional[str], semesters: int = 8) -> Optional[bool]:
    per = _parse_number(per_sem_str)
    tot = _parse_number(total_str)
    if per is None or tot is None:
        return None
    # Accept small rounding differences
    return abs(tot - (per * semesters)) < 1e-3


def _gather_sources(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_undergraduate_checks(
    evaluator: Evaluator,
    parent,
    ug: UndergraduateInfo,
    refs: ReferenceURLs
) -> None:
    node = evaluator.add_parallel(
        id="Undergraduate_Selection_and_Cost",
        desc="Select an eligible NC Promise undergraduate institution/program and provide tuition figures and 8-semester total tuition calculation.",
        parent=parent,
        critical=True
    )

    # 1) University name provided
    evaluator.add_custom_node(
        result=bool(ug and ug.university_name and ug.university_name.strip()),
        id="Undergraduate_University_Name_Provided",
        desc="Provides the name of the NC Promise university selected for undergraduate study (requested output).",
        parent=node,
        critical=True
    )

    # 2) University is among allowed four
    evaluator.add_custom_node(
        result=_is_nc_promise_school(ug.university_name),
        id="NC_Promise_University_Is_One_of_Allowed_Four",
        desc="Undergraduate university is one of: Elizabeth City State University, Fayetteville State University, University of North Carolina at Pembroke, or Western Carolina University.",
        parent=node,
        critical=True
    )

    # 3) Program is education/teacher prep leading to licensure (source-grounded)
    leaf_prog = evaluator.add_leaf(
        id="Undergraduate_Program_Is_Education_or_Teacher_Prep_Leading_to_Licensure",
        desc="Identifies a bachelor's program in education/teacher preparation that leads to teaching licensure.",
        parent=node,
        critical=True
    )
    prog_claim = "This undergraduate bachelor's program is in education or teacher preparation and leads to initial K-12 teaching licensure."
    await evaluator.verify(
        claim=prog_claim,
        node=leaf_prog,
        sources=refs.undergrad_url,
        additional_instruction="Confirm the page clearly indicates an education/teacher preparation bachelor's program that leads to initial teacher licensure."
    )

    # 4) Undergraduate institution regionally accredited (answer-level assertion)
    evaluator.add_custom_node(
        result=_string_truthy(ug.is_regionally_accredited),
        id="Undergraduate_University_Is_Regionally_Accredited",
        desc="States the undergraduate institution is regionally accredited (per constraints).",
        parent=node,
        critical=True
    )

    # 5) NC Promise tuition per semester provided with eligible year (2025–26 or 2026–27)
    evaluator.add_custom_node(
        result=bool(ug and ug.nc_promise_tuition_per_semester and _year_is_eligible(ug.tuition_academic_year)),
        id="NC_Promise_Tuition_Rate_Per_Semester_Provided_With_Eligible_Year",
        desc="Provides the in-state NC Promise tuition rate per semester and specifies it is for the 2025–2026 or 2026–2027 academic year (requested output).",
        parent=node,
        critical=True
    )

    # 6) If using 2025–26, per-semester rate must be $500
    if _year_is_2025_26(ug.tuition_academic_year):
        evaluator.add_custom_node(
            result=(_parse_number(ug.nc_promise_tuition_per_semester) == 500.0),
            id="NC_Promise_Tuition_Rate_Matches_Constraint_When_Using_2025_26",
            desc="If the answer uses academic year 2025–2026, the in-state NC Promise tuition rate per semester is $500 (per constraints).",
            parent=node,
            critical=True
        )

    # 7) Total undergraduate tuition for 8 semesters provided and correct
    calc_ok = _calc_total_ok(ug.nc_promise_tuition_per_semester, ug.eight_semester_total, semesters=8)
    evaluator.add_custom_node(
        result=bool(calc_ok),
        id="Total_Undergraduate_Tuition_For_8_Semesters_Provided_And_Correct",
        desc="Provides and correctly calculates the total undergraduate tuition cost for 8 semesters using the stated per-semester NC Promise tuition rate (requested output).",
        parent=node,
        critical=True
    )


async def build_teaching_checks(
    evaluator: Evaluator,
    parent,
    teach: TeachingInfo,
    refs: ReferenceURLs
) -> None:
    node = evaluator.add_parallel(
        id="Teaching_License_and_Experience",
        desc="States the teaching license step and the required post-licensure K-12 teaching experience before pursuing principal licensure.",
        parent=parent,
        critical=True
    )

    # 1) Teaching license obtained after bachelor's (source-grounded)
    leaf_license = evaluator.add_leaf(
        id="Teaching_License_Obtained_After_Bachelors",
        desc="States that a valid teaching license is obtained after completing the bachelor's degree (per constraints).",
        parent=node,
        critical=True
    )
    license_claim = (
        "After completing a bachelor's degree and an approved teacher preparation program, a candidate obtains a valid North Carolina teaching license."
    )
    await evaluator.verify(
        claim=license_claim,
        node=leaf_license,
        sources=_gather_sources(refs.teaching_url, refs.licensure_url),
        additional_instruction="Verify the official page indicates that completing a bachelor's degree and an approved teacher preparation program leads to a NC teaching license."
    )

    # 2) Post-licensure K-12 teaching experience ≥ 3 years (answer-level numeric constraint)
    years_val = _parse_number(teach.post_licensure_experience_years if teach else None)
    evaluator.add_custom_node(
        result=(years_val is not None and years_val >= 3.0),
        id="Post_Licensure_K12_Teaching_Experience_Min_3_Years",
        desc="Specifies a minimum of 3 years of post-licensure teaching experience in a K-12 classroom setting (per prompt/constraints).",
        parent=node,
        critical=True
    )

    # 3) Teaching experience duration provided in years (existence)
    evaluator.add_custom_node(
        result=(years_val is not None and years_val > 0),
        id="Teaching_Experience_Duration_Provided_In_Years",
        desc="Explicitly provides the teaching experience duration in years (requested output).",
        parent=node,
        critical=True
    )


async def build_graduate_checks(
    evaluator: Evaluator,
    parent,
    grad: GraduateInfo,
    refs: ReferenceURLs
) -> None:
    node = evaluator.add_parallel(
        id="Graduate_Program_and_Licensure_Eligibility",
        desc="Identify an eligible graduate program and provide program attributes tied to principal/administrator licensure eligibility.",
        parent=parent,
        critical=False  # To allow non-critical 'typical' checks below
    )

    # 1) Graduate university is regionally accredited (answer-level assertion)
    evaluator.add_custom_node(
        result=_string_truthy(grad.is_regionally_accredited if grad else None),
        id="Graduate_University_Is_Regionally_Accredited",
        desc="States the graduate institution is regionally accredited (per constraints).",
        parent=node,
        critical=True
    )

    # 2) Graduate program name and type provided (existence)
    evaluator.add_custom_node(
        result=bool(grad and grad.program_name_and_type and grad.program_name_and_type.strip()),
        id="Graduate_Program_Name_and_Type_Provided",
        desc="Provides the name and type of the graduate program in educational leadership/administration (requested output).",
        parent=node,
        critical=True
    )

    # 3) Graduate program is master's level (source-grounded)
    leaf_master = evaluator.add_leaf(
        id="Graduate_Program_Is_Masters_Level",
        desc="Graduate program leads to a master's degree (e.g., M.Ed., M.A., M.S., or Ed.S.) (per constraints).",
        parent=node,
        critical=True
    )
    master_claim = "This graduate program is a master's-level degree (e.g., M.Ed., M.A., M.S., Ed.S.)."
    await evaluator.verify(
        claim=master_claim,
        node=leaf_master,
        sources=refs.graduate_url,
        additional_instruction="Confirm the program page shows a master's-level degree type (e.g., M.Ed., M.A., M.S., or Ed.S.)."
    )

    # 4) Field is Educational Leadership or Educational Administration (source-grounded)
    leaf_field = evaluator.add_leaf(
        id="Graduate_Program_Field_Is_Ed_Leadership_or_Ed_Administration",
        desc="Graduate program is specifically in educational leadership or educational administration (per prompt/constraints).",
        parent=node,
        critical=True
    )
    field_claim = "This program is in the field of Educational Leadership or Educational Administration."
    await evaluator.verify(
        claim=field_claim,
        node=leaf_field,
        sources=refs.graduate_url,
        additional_instruction="Check program title/description to ensure it is in Educational Leadership or Educational Administration."
    )

    # 5) Program leads to principal/administrator licensure eligibility (source-grounded)
    leaf_lic_elig = evaluator.add_leaf(
        id="Program_Leads_to_Principal_Administrator_Licensure_Eligibility",
        desc="Graduate program includes or leads to eligibility for principal/administrator licensure (per prompt/constraints).",
        parent=node,
        critical=True
    )
    lic_elig_claim = "This graduate program includes or leads to eligibility for principal/administrator licensure."
    await evaluator.verify(
        claim=lic_elig_claim,
        node=leaf_lic_elig,
        sources=_gather_sources(refs.graduate_url, refs.licensure_url),
        additional_instruction="Verify the program page or licensure page indicates principal/administrator licensure eligibility from this program."
    )

    # 6) Admission requirements include minimum GPA value (source-grounded)
    leaf_min_gpa = evaluator.add_leaf(
        id="Admission_Requirements_Include_Minimum_GPA_Value",
        desc="Graduate program admission requirements clearly state a minimum GPA and the minimum GPA value is provided (per prompt/constraints; requested output).",
        parent=node,
        critical=True
    )
    gpa_val_str = grad.admission_min_gpa if grad else None
    gpa_claim = f"The minimum GPA for admission to this program is {gpa_val_str}." if gpa_val_str else "The program lists a minimum GPA requirement for admission."
    await evaluator.verify(
        claim=gpa_claim,
        node=leaf_min_gpa,
        sources=refs.graduate_url,
        additional_instruction="Find the explicit minimum GPA requirement stated in the admissions requirements on the program page."
    )

    # 7) Graduate credit hours total provided (source-grounded)
    leaf_credits = evaluator.add_leaf(
        id="Graduate_Credit_Hours_Total_Provided",
        desc="Provides the total credit hours required for the graduate degree (requested output).",
        parent=node,
        critical=True
    )
    credits_val_str = grad.total_credit_hours if grad else None
    credits_claim = f"The total credit hours required for this graduate degree is {credits_val_str}." if credits_val_str else "The total credit hours required for the program is specified."
    await evaluator.verify(
        claim=credits_claim,
        node=leaf_credits,
        sources=refs.graduate_url,
        additional_instruction="Identify the total credit hours requirement on the program page."
    )

    # 8) Minimum GPA ≥ 3.0 (typical, non-critical) - answer-level numeric check
    gpa_ge_3 = _gpa_at_least_min(gpa_val_str, 3.0)
    evaluator.add_custom_node(
        result=(gpa_ge_3 is True),
        id="Minimum_GPA_Is_At_Least_3_0_Typical",
        desc="Minimum GPA is ≥ 3.0 (noted as typical in constraints).",
        parent=node,
        critical=False
    )

    # 9) Credit hours within 30–36 (typical, non-critical) - answer-level numeric check
    credits_typical = _credits_in_typical_range(credits_val_str)
    evaluator.add_custom_node(
        result=(credits_typical is True),
        id="Graduate_Credit_Hours_In_30_to_36_Range_Typical",
        desc="Graduate credit hours fall within 30–36 (noted as typical in constraints).",
        parent=node,
        critical=False
    )


async def build_reference_url_checks(
    evaluator: Evaluator,
    parent,
    refs: ReferenceURLs
) -> None:
    node = evaluator.add_parallel(
        id="Required_Reference_URLs",
        desc="Provide official-source reference URLs for each major component requested by the prompt.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(refs and refs.undergrad_url and refs.undergrad_url.strip()),
        id="URL_Undergraduate_Program",
        desc="Provides an official-source reference URL for the undergraduate program/university component.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(refs and refs.teaching_url and refs.teaching_url.strip()),
        id="URL_Teaching_Requirements",
        desc="Provides an official-source reference URL for the teaching license and/or teaching experience requirement component.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(refs and refs.graduate_url and refs.graduate_url.strip()),
        id="URL_Graduate_Program",
        desc="Provides an official-source reference URL for the graduate program component.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(refs and refs.licensure_url and refs.licensure_url.strip()),
        id="URL_Licensure_Requirements",
        desc="Provides an official-source reference URL for the principal/administrator licensure requirements component.",
        parent=node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the NC Promise principal licensure pathway task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Sequential: later steps skipped if earlier essential steps fail
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

    # Extract structured information
    extracted: PathwayExtraction = await evaluator.extract(
        prompt=prompt_extract_pathway(),
        template_class=PathwayExtraction,
        extraction_name="pathway_extraction"
    )

    # Build the top-level "Complete_Educational_Pathway" node (non-critical to allow nuanced grading;
    # essential failures still enforced by critical subtrees)
    top = evaluator.add_sequential(
        id="Complete_Educational_Pathway",
        desc="Provide a complete pathway (undergrad at an eligible NC Promise school/program → teaching license → ≥3 years post-licensure K-12 teaching → graduate leadership/admin program leading to principal/administrator licensure eligibility) and all requested outputs with official reference URLs.",
        parent=root,
        critical=False
    )

    ug = extracted.undergraduate or UndergraduateInfo()
    teach = extracted.teaching or TeachingInfo()
    grad = extracted.graduate or GraduateInfo()
    refs = extracted.references or ReferenceURLs()

    # Undergraduate checks
    await build_undergraduate_checks(evaluator, top, ug, refs)

    # Teaching checks
    await build_teaching_checks(evaluator, top, teach, refs)

    # Graduate checks
    await build_graduate_checks(evaluator, top, grad, refs)

    # Reference URLs checks
    await build_reference_url_checks(evaluator, top, refs)

    # Add useful custom info for debugging
    evaluator.add_custom_info(
        info={
            "allowed_nc_promise_schools": sorted(list(ALLOWED_NC_PROMISE_SCHOOLS)),
            "eligible_academic_years_examples": sorted(list(ELIGIBLE_AY_LABELS)),
            "extracted_snapshot": {
                "undergraduate": ug.dict(),
                "teaching": teach.dict(),
                "graduate": grad.dict(),
                "references": refs.dict(),
            },
        },
        info_type="context",
        info_name="evaluation_context"
    )

    return evaluator.get_summary()