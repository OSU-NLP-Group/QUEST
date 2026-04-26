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
TASK_ID = "pa_cacrep_school_counseling_near_phl"
TASK_DESCRIPTION = """
Identify a CACREP-accredited school counseling graduate program offered by a university in Pennsylvania that is located within 15 miles of Philadelphia and leads to Pennsylvania PK-12 School Counselor certification. Provide the following information:

1. The name of the university and its location (city and distance from Philadelphia)
2. The specific name and degree type of the school counseling program
3. Confirmation of CACREP accreditation status with a reference URL
4. The total credit hours required for the program
5. Description of the field experience requirements (practicum and internship), including both elementary and secondary level placements
6. Confirmation that the program leads to Pennsylvania PK-12 School Counselor certification
7. The prerequisite professional experience requirement for PA school counselor certification
8. The certification examination requirement for PA school counselor certification

For each piece of information provided, include the reference URL(s) from the university's official website or the CACREP directory where you found this information.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramExtraction(BaseModel):
    # University/location
    university_name: Optional[str] = None
    city: Optional[str] = None
    distance_miles: Optional[str] = None  # e.g., "8 miles", "10 mi", "within 15 miles"
    location_urls: List[str] = Field(default_factory=list)

    # Program + degree
    program_name: Optional[str] = None
    degree_type: Optional[str] = None  # e.g., MS, MA, MEd, MS.Ed.
    program_urls: List[str] = Field(default_factory=list)

    # CACREP accreditation
    cacrep_accredited_text: Optional[str] = None
    cacrep_urls: List[str] = Field(default_factory=list)

    # Credits
    total_credits: Optional[str] = None  # keep as string for flexibility (e.g., "60 credits")
    credit_urls: List[str] = Field(default_factory=list)

    # CACREP core curriculum coverage (8 core areas)
    core_coverage_text: Optional[str] = None
    core_coverage_urls: List[str] = Field(default_factory=list)

    # Field experiences
    practicum_text: Optional[str] = None
    internship_text: Optional[str] = None
    supervision_text: Optional[str] = None  # mentions "supervised"
    elementary_secondary_text: Optional[str] = None  # mentions placements at both levels
    field_urls: List[str] = Field(default_factory=list)

    # Leads to PA PK-12 certification
    leads_to_pa_pk12_text: Optional[str] = None
    lead_urls: List[str] = Field(default_factory=list)

    # PA prerequisite professional experience
    pa_experience_requirement_text: Optional[str] = None
    experience_urls: List[str] = Field(default_factory=list)

    # PA exam requirement
    pa_exam_requirement_text: Optional[str] = None
    exam_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program_info() -> str:
    return """
Extract the single target program and its supporting URLs exactly as presented in the answer.

Return a JSON object with the following fields. If a field is missing from the answer, return null for the field or an empty list for URL arrays. Do NOT invent any information. Extract only from the answer text.

Required fields:
- university_name: The university's name
- city: The city where the university/campus is located (as stated in the answer)
- distance_miles: The numeric distance (in miles) from Philadelphia as stated in the answer (e.g., "8 miles", "10 mi", "within 15 miles"). If not explicitly numeric, return the phrase used.
- location_urls: All URLs cited for location/city/distance. Prefer official university (.edu) or CACREP (cacrep.org) URLs.

- program_name: The exact school counseling program name/title
- degree_type: The degree type (e.g., "MS", "MA", "MEd", "MS.Ed.", "M.S.Ed.")
- program_urls: URLs cited for the program page/overview/handbook/catalog (ideally on the university site or CACREP)

- cacrep_accredited_text: The wording in the answer that asserts CACREP accreditation
- cacrep_urls: URLs cited to support CACREP accreditation (CACREP directory and/or official university page)

- total_credits: The stated total graduate credit hours required for the program (e.g., "60 credits")
- credit_urls: URLs cited to support total credit hours (catalog, handbook, program page, or CACREP)

- core_coverage_text: The wording that indicates coverage of the 8 CACREP core curriculum areas (explicit list or mapping statement)
- core_coverage_urls: URLs cited to support coverage of CACREP core areas (program curriculum page, handbook, CACREP listing)

- practicum_text: The wording that describes practicum requirement
- internship_text: The wording that describes internship requirement
- supervision_text: The wording that indicates these field experiences are supervised
- elementary_secondary_text: The wording that indicates placements include both elementary (PreK-8) and secondary (7-12)
- field_urls: URLs cited for practicum/internship/placements

- leads_to_pa_pk12_text: The wording that completing the program leads to eligibility for PA PK-12 School Counselor certification
- lead_urls: URLs cited for this (university certification page, program page, or CACREP)

- pa_experience_requirement_text: The wording that PA requires 2 years of prior professional experience (teaching, social work, or professional counseling)
- experience_urls: URLs cited for this requirement (prefer university or CACREP if provided)

- pa_exam_requirement_text: The wording that graduates must pass the Professional School Counselor exam (ETS/PRAXIS)
- exam_urls: URLs cited for this exam requirement (prefer university or CACREP if provided)

Notes and rules:
- Only include URLs that appear in the answer text. Do not infer or add URLs that aren't cited.
- When multiple URLs are provided for the same item, include all of them.
- Keep text fields exactly as stated in the answer (do not normalize).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _join_sources(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for u in lst or []:
            if not isinstance(u, str):
                continue
            if u not in seen and u.strip():
                seen.add(u)
                out.append(u)
    return out


def _is_allowed_source(url: str) -> bool:
    if not isinstance(url, str):
        return False
    u = url.lower()
    return (".edu/" in u) or u.endswith(".edu") or ("cacrep.org" in u)


def _has_allowed_url(urls: List[str]) -> bool:
    return any(_is_allowed_source(u) for u in urls or [])


def _allowed_source_instruction() -> str:
    return (
        "Only consider this claim supported if at least one provided URL is from either: "
        "(a) the university's official .edu domain, or (b) the official CACREP directory at cacrep.org. "
        "If the provided URL(s) are not .edu or cacrep.org, mark the verification as not supported."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_single_program_identified(evaluator: Evaluator, parent) -> None:
    node = evaluator.add_leaf(
        id="single_program_identified",
        desc="Response identifies one (and only one) university + one specific graduate school counseling program as the target program.",
        parent=parent,
        critical=True,
    )
    claim = (
        "The response identifies exactly one university and exactly one specific graduate school counseling graduate program "
        "as the sole target program (no multiple choices, no lists of alternatives)."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="If multiple programs or multiple universities are listed as options, or more than one target is given, mark this incorrect."
    )


async def build_university_and_geography(evaluator: Evaluator, parent, data: ProgramExtraction) -> None:
    grp = evaluator.add_parallel(
        id="university_and_geography",
        desc="University location and proximity constraints are satisfied and reported with allowed citations.",
        parent=parent,
        critical=True,
    )

    # University is in Pennsylvania
    n_pa = evaluator.add_leaf(
        id="university_in_pennsylvania",
        desc="University is located in Pennsylvania (as stated in the response).",
        parent=grp,
        critical=True,
    )
    claim_pa = f"The university '{data.university_name or 'the identified university'}' is located in the state of Pennsylvania (PA)."
    await evaluator.verify(
        claim=claim_pa,
        node=n_pa,
        sources=_join_sources(data.location_urls, data.cacrep_urls, data.program_urls),
        additional_instruction=_allowed_source_instruction() + " Accept if the official page shows address or city/state in PA."
    )

    # Within 15 miles of Philadelphia (numeric stated in the answer)
    n_within = evaluator.add_leaf(
        id="within_15_miles_of_philadelphia",
        desc="Response states the university is within 15 miles of Philadelphia (numeric distance provided).",
        parent=grp,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly provides a numeric distance to Philadelphia that is 15 miles or fewer.",
        node=n_within,
        additional_instruction="Look for a number in miles in the answer text itself (e.g., '8 miles from Philadelphia'). If not explicitly numeric, mark incorrect."
    )

    # City reported
    n_city = evaluator.add_leaf(
        id="city_reported",
        desc="Response provides the university location city (as requested).",
        parent=grp,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states the university's location city.",
        node=n_city,
        additional_instruction="Passing requires that the city name is present in the answer text."
    )

    # Location/distance citations allowed (support + allowed domains)
    n_cite = evaluator.add_leaf(
        id="location_and_distance_citations_allowed",
        desc="Provides reference URL(s) supporting the city/location and the distance-to-Philadelphia claim; URLs must be from the university's official website or the CACREP directory.",
        parent=grp,
        critical=True,
    )
    claim_cite = (
        "The provided URL(s) support the stated city/location for the university and the proximity to Philadelphia "
        "(e.g., campus in Philadelphia or an official statement indicating the location)."
    )
    await evaluator.verify(
        claim=claim_cite,
        node=n_cite,
        sources=_join_sources(data.location_urls, data.program_urls, data.cacrep_urls),
        additional_instruction=_allowed_source_instruction() +
        " If the campus is in Philadelphia proper, that is inherently within 15 miles."
    )


async def build_program_name_and_degree(evaluator: Evaluator, parent, data: ProgramExtraction) -> None:
    grp = evaluator.add_parallel(
        id="program_name_and_degree",
        desc="Program name and degree type are provided and meet degree-level constraint, with allowed citations.",
        parent=parent,
        critical=True,
    )

    n_name = evaluator.add_leaf(
        id="program_name_provided",
        desc="Response includes the specific school counseling program name/title.",
        parent=grp,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes the specific title/name of the school counseling graduate program (not just a vague label).",
        node=n_name
    )

    n_degree = evaluator.add_leaf(
        id="degree_type_provided",
        desc="Response includes the degree type (e.g., MS/MA/MEd).",
        parent=grp,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes the degree type for the program (e.g., MS, MA, MEd, MS.Ed., M.S.Ed., or similar).",
        node=n_degree
    )

    n_masters = evaluator.add_leaf(
        id="meets_masters_degree_requirement",
        desc="Degree is a master's degree program (satisfies the listed constraint that PA certification requires completion of a master's degree program).",
        parent=grp,
        critical=True,
    )
    await evaluator.verify(
        claim="The stated degree is a master's level degree (e.g., MA, MS, MEd, MS.Ed., M.S.Ed.).",
        node=n_masters,
        additional_instruction="If the degree is post-master's certificate, doctoral, or non-master's, mark incorrect."
    )

    n_prog_cite = evaluator.add_leaf(
        id="program_citations_allowed",
        desc="Provides reference URL(s) supporting program name and degree type; URLs must be from the university's official website or the CACREP directory.",
        parent=grp,
        critical=True,
    )
    claim_prog_cite = (
        "The provided URL(s) substantiate the program's official name and degree type as stated in the answer."
    )
    await evaluator.verify(
        claim=claim_prog_cite,
        node=n_prog_cite,
        sources=_join_sources(data.program_urls, data.cacrep_urls),
        additional_instruction=_allowed_source_instruction()
    )


async def build_cacrep_accreditation(evaluator: Evaluator, parent, data: ProgramExtraction) -> None:
    grp = evaluator.add_parallel(
        id="cacrep_accreditation",
        desc="CACREP accreditation is confirmed with allowed citations.",
        parent=parent,
        critical=True,
    )

    n_status = evaluator.add_leaf(
        id="cacrep_status_confirmed",
        desc="Response confirms the program is CACREP-accredited (as stated by CACREP and/or the university).",
        parent=grp,
        critical=True,
    )
    claim_status = (
        "The named school counseling program is CACREP-accredited (School Counseling specialization)."
    )
    await evaluator.verify(
        claim=claim_status,
        node=n_status,
        sources=_join_sources(data.cacrep_urls, data.program_urls),
        additional_instruction=_allowed_source_instruction() + " Prefer confirmation from cacrep.org directory; an official university accreditation page also suffices if current."
    )

    n_url_present = evaluator.add_custom_node(
        result=_has_allowed_url(_join_sources(data.cacrep_urls, data.program_urls)),
        id="cacrep_accreditation_reference_url_present",
        desc="Includes at least one reference URL that supports the CACREP accreditation claim; URL must be from the university's official website or the CACREP directory.",
        parent=grp,
        critical=True
    )


async def build_credit_hours(evaluator: Evaluator, parent, data: ProgramExtraction) -> None:
    grp = evaluator.add_parallel(
        id="credit_hours_requirement",
        desc="Total credit hours are provided and satisfy the listed CACREP credit-hour constraint, with allowed citations.",
        parent=parent,
        critical=True,
    )

    n_provided = evaluator.add_custom_node(
        result=bool(data.total_credits and str(data.total_credits).strip()),
        id="total_credit_hours_provided",
        desc="Response states the program's total required graduate credit hours.",
        parent=grp,
        critical=True
    )

    n_eq60 = evaluator.add_leaf(
        id="credit_hours_equals_60",
        desc="Total required graduate credit hours are 60 (per the listed constraint: CACREP-accredited programs require 60 graduate credit hours).",
        parent=grp,
        critical=True,
    )
    await evaluator.verify(
        claim="The program requires 60 graduate credit hours to complete.",
        node=n_eq60,
        sources=_join_sources(data.credit_urls, data.program_urls, data.cacrep_urls),
        additional_instruction=_allowed_source_instruction() + " Check catalog/handbook/program page for a '60 credits' indication."
    )

    n_cite_allowed = evaluator.add_custom_node(
        result=_has_allowed_url(_join_sources(data.credit_urls, data.program_urls, data.cacrep_urls)),
        id="credit_hours_citations_allowed",
        desc="Provides reference URL(s) supporting the total credit hours; URLs must be from the university's official website or the CACREP directory.",
        parent=grp,
        critical=True
    )


async def build_core_curriculum(evaluator: Evaluator, parent, data: ProgramExtraction) -> None:
    grp = evaluator.add_parallel(
        id="cacrep_core_curriculum_coverage",
        desc="Program indicates coverage of CACREP core curriculum areas, with allowed citations.",
        parent=parent,
        critical=True,
    )

    n_cover = evaluator.add_leaf(
        id="covers_all_8_cacrep_core_areas",
        desc="Response provides confirmation (from a cited source) that the program includes coursework in the 8 CACREP core curriculum areas (may be stated explicitly or via curriculum mapping).",
        parent=grp,
        critical=True,
    )
    await evaluator.verify(
        claim="The program includes coursework covering the eight CACREP core curriculum areas (explicit statement or curriculum mapping indicating all 8).",
        node=n_cover,
        sources=_join_sources(data.core_coverage_urls, data.program_urls, data.cacrep_urls),
        additional_instruction=_allowed_source_instruction()
    )

    n_cite_allowed = evaluator.add_custom_node(
        result=_has_allowed_url(_join_sources(data.core_coverage_urls, data.program_urls, data.cacrep_urls)),
        id="core_areas_citations_allowed",
        desc="Provides reference URL(s) supporting the CACREP core curriculum coverage claim; URLs must be from the university's official website or the CACREP directory.",
        parent=grp,
        critical=True
    )


async def build_field_experiences(evaluator: Evaluator, parent, data: ProgramExtraction) -> None:
    grp = evaluator.add_parallel(
        id="field_experiences",
        desc="Field experience requirements are described and meet listed constraints, with allowed citations.",
        parent=parent,
        critical=True,
    )

    n_prac = evaluator.add_leaf(
        id="practicum_requirement_described",
        desc="Response describes the practicum requirement (as part of field experience requirements).",
        parent=grp,
        critical=True,
    )
    await evaluator.verify(
        claim="The program requires a practicum as part of its field experience requirements.",
        node=n_prac,
        sources=_join_sources(data.field_urls, data.program_urls),
        additional_instruction=_allowed_source_instruction()
    )

    n_intern = evaluator.add_leaf(
        id="internship_requirement_described",
        desc="Response describes the internship requirement (as part of field experience requirements).",
        parent=grp,
        critical=True,
    )
    await evaluator.verify(
        claim="The program requires an internship as part of its field experience requirements.",
        node=n_intern,
        sources=_join_sources(data.field_urls, data.program_urls),
        additional_instruction=_allowed_source_instruction()
    )

    n_supervised = evaluator.add_leaf(
        id="supervised_practicum_and_internship_present",
        desc="Response confirms the practicum and internship are supervised (per listed constraint).",
        parent=grp,
        critical=True,
    )
    await evaluator.verify(
        claim="Both the practicum and the internship are supervised experiences.",
        node=n_supervised,
        sources=_join_sources(data.field_urls, data.program_urls),
        additional_instruction=_allowed_source_instruction()
    )

    n_levels = evaluator.add_leaf(
        id="elementary_and_secondary_placements_confirmed",
        desc="Response confirms field experiences include both elementary (PreK-8) and secondary (7-12) placements (per listed constraint).",
        parent=grp,
        critical=True,
    )
    await evaluator.verify(
        claim="Field experiences include placements at both the elementary (PreK–8) and secondary (7–12) levels.",
        node=n_levels,
        sources=_join_sources(data.field_urls, data.program_urls),
        additional_instruction=_allowed_source_instruction()
    )

    n_cite_allowed = evaluator.add_custom_node(
        result=_has_allowed_url(_join_sources(data.field_urls, data.program_urls)),
        id="field_experience_citations_allowed",
        desc="Provides reference URL(s) supporting practicum/internship requirements and elementary/secondary placement expectation; URLs must be from the university's official website or the CACREP directory.",
        parent=grp,
        critical=True
    )


async def build_leads_to_pa_cert(evaluator: Evaluator, parent, data: ProgramExtraction) -> None:
    grp = evaluator.add_parallel(
        id="leads_to_pa_pk12_certification",
        desc="Program leads to PA PK-12 School Counselor certification, with allowed citations.",
        parent=parent,
        critical=True,
    )

    n_leads = evaluator.add_leaf(
        id="pa_pk12_certification_lead_confirmed",
        desc="Response confirms completing the program leads to eligibility for Pennsylvania PK-12 School Counselor certification.",
        parent=grp,
        critical=True,
    )
    await evaluator.verify(
        claim="Completing this program leads to eligibility for Pennsylvania PK–12 School Counselor certification.",
        node=n_leads,
        sources=_join_sources(data.lead_urls, data.program_urls, data.cacrep_urls),
        additional_instruction=_allowed_source_instruction()
    )

    n_cite_allowed = evaluator.add_custom_node(
        result=_has_allowed_url(_join_sources(data.lead_urls, data.program_urls, data.cacrep_urls)),
        id="pa_certification_citations_allowed",
        desc="Provides reference URL(s) supporting the program-to-PA-certification claim; URLs must be from the university's official website or the CACREP directory.",
        parent=grp,
        critical=True
    )


async def build_pa_experience_req(evaluator: Evaluator, parent, data: ProgramExtraction) -> None:
    grp = evaluator.add_parallel(
        id="pa_certification_prerequisite_experience",
        desc="PA prerequisite professional experience requirement is stated per listed constraint, with allowed citations.",
        parent=parent,
        critical=True,
    )

    n_two_years = evaluator.add_leaf(
        id="two_years_experience_requirement_stated",
        desc="Response states PA requires 2 years of prior professional experience (teaching, social work, or professional counseling) (per listed constraint).",
        parent=grp,
        critical=True,
    )
    await evaluator.verify(
        claim="Pennsylvania requires 2 years of prior professional experience (teaching, social work, or professional counseling) for School Counselor certification.",
        node=n_two_years,
        sources=_join_sources(data.experience_urls, data.program_urls),
        additional_instruction=_allowed_source_instruction()
    )

    n_cite_allowed = evaluator.add_custom_node(
        result=_has_allowed_url(_join_sources(data.experience_urls, data.program_urls)),
        id="experience_requirement_citations_allowed",
        desc="Provides reference URL(s) supporting the PA experience requirement; URLs must be from the university's official website or the CACREP directory.",
        parent=grp,
        critical=True
    )


async def build_pa_exam_req(evaluator: Evaluator, parent, data: ProgramExtraction) -> None:
    grp = evaluator.add_parallel(
        id="pa_certification_exam_requirement",
        desc="PA certification exam requirement is stated per listed constraint, with allowed citations.",
        parent=parent,
        critical=True,
    )

    n_exam = evaluator.add_leaf(
        id="professional_school_counselor_exam_stated",
        desc="Response states graduates must pass the Professional School Counselor Exam (ETS/PRAXIS) (per listed constraint).",
        parent=grp,
        critical=True,
    )
    await evaluator.verify(
        claim="Graduates must pass the Professional School Counselor exam (ETS/PRAXIS) for Pennsylvania School Counselor certification.",
        node=n_exam,
        sources=_join_sources(data.exam_urls, data.program_urls),
        additional_instruction=_allowed_source_instruction() + " Accept equivalents like 'Praxis Professional School Counselor (5421/5422)'."
    )

    n_cite_allowed = evaluator.add_custom_node(
        result=_has_allowed_url(_join_sources(data.exam_urls, data.program_urls)),
        id="exam_requirement_citations_allowed",
        desc="Provides reference URL(s) supporting the PA exam requirement; URLs must be from the university's official website or the CACREP directory.",
        parent=grp,
        critical=True
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
    Evaluate an answer for the PA CACREP-accredited school counseling near Philadelphia task.
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
        default_model=model,
    )
    # Set root as critical to match rubric (ensure to do this before adding any children)
    root.critical = True

    # 1) Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_program_info(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction"
    )

    # 2) Build verification tree according to rubric
    await build_single_program_identified(evaluator, root)
    await build_university_and_geography(evaluator, root, extracted)
    await build_program_name_and_degree(evaluator, root, extracted)
    await build_cacrep_accreditation(evaluator, root, extracted)
    await build_credit_hours(evaluator, root, extracted)
    await build_core_curriculum(evaluator, root, extracted)
    await build_field_experiences(evaluator, root, extracted)
    await build_leads_to_pa_cert(evaluator, root, extracted)
    await build_pa_experience_req(evaluator, root, extracted)
    await build_pa_exam_req(evaluator, root, extracted)

    # 3) Return structured summary
    return evaluator.get_summary()