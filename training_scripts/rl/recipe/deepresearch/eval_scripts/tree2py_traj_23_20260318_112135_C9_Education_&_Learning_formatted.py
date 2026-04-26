import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "big_ten_public_eng_universities_1848_1870"
TASK_DESCRIPTION = """
Identify three public universities that are current members of the Big Ten Conference (as of 2026) and were founded between 1848 and 1870 (inclusive). For each university, you must provide: (1) The university's official name, (2) The exact year it was founded, (3) Verification that it is a public institution and a Big Ten Conference member, (4) Current total student enrollment (must exceed 40,000 students), (5) The name of the university's College or School of Engineering, (6) At least one ABET-accredited undergraduate engineering program in Computer Engineering, Electrical Engineering, or Mechanical Engineering, (7) Confirmation that bachelor's degree programs require a minimum of 120 credit hours, (8) Evidence of on-campus residence halls (dormitories) for undergraduate students, (9) Evidence of campus dining facilities with meal plan options, (10) Evidence of a student recreation center or fitness facility, (11) Evidence of a main library or library system on campus, (12) The name and description of the university's International Student Services office or equivalent, (13) The minimum TOEFL and/or IELTS scores required for international undergraduate student admission. For each piece of information provided, include a reference URL from an official university website or authoritative source that verifies the claim.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    # Identity and membership
    name: Optional[str] = None
    founded_year: Optional[str] = None  # Keep as string to be lenient with formats like "1867"
    identity_urls: List[str] = Field(default_factory=list)  # official/about/bigten/athletics pages that support identity + membership + public status

    # Enrollment
    total_enrollment: Optional[str] = None  # e.g., "52,000", "over 45,000"
    has_undergraduates: Optional[bool] = None
    enrollment_urls: List[str] = Field(default_factory=list)  # IR/Fact Book/Enrollment snapshot pages

    # Engineering
    engineering_college_name: Optional[str] = None
    abet_program_name: Optional[str] = None  # e.g., "Mechanical Engineering (BSME)"
    abet_program_field: Optional[str] = None  # One of: "Computer Engineering", "Electrical Engineering", "Mechanical Engineering"
    credit_hours_min: Optional[str] = None  # e.g., "120", "≥120"
    engineering_urls: List[str] = Field(default_factory=list)  # engineering site, ABET page, program page, catalog

    # Campus infrastructure
    residence_urls: List[str] = Field(default_factory=list)    # housing/residence halls pages
    dining_urls: List[str] = Field(default_factory=list)       # dining/meal plan pages
    recreation_urls: List[str] = Field(default_factory=list)   # campus rec/fitness center pages
    library_urls: List[str] = Field(default_factory=list)      # main library/library system pages
    infrastructure_urls: List[str] = Field(default_factory=list)  # any additional infra URLs if provided

    # International services
    international_office_name: Optional[str] = None
    international_office_desc: Optional[str] = None
    toefl_min: Optional[str] = None
    ielts_min: Optional[str] = None
    international_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to three universities presented in the answer that are intended to satisfy the task requirements.
    For each university, return a JSON object with the following fields (use null where information is missing):

    1) name: The university's official name exactly as stated in the answer.
    2) founded_year: The exact founding year mentioned in the answer (e.g., "1867"). If a range or multiple dates are given, use the primary "founded/established" year used in the answer.
    3) identity_urls: Array of URLs (official or authoritative) that support the university identity, public status, and Big Ten membership as of 2026. Good sources: university official sites (.edu), the Big Ten website, or authoritative pages.
    4) total_enrollment: The total current enrollment figure in the answer text (string, accept phrases like "over 50,000").
    5) has_undergraduates: true/false if the answer explicitly indicates the university enrolls undergraduates; null if not stated.
    6) enrollment_urls: Array of URLs that verify total enrollment and/or undergraduate enrollment.
    7) engineering_college_name: The exact name of the College/School of Engineering as stated.
    8) abet_program_name: The specific ABET-accredited undergraduate program name in Computer Engineering, Electrical Engineering, or Mechanical Engineering (e.g., "Mechanical Engineering (B.S.)"). If multiple are cited, pick one.
    9) abet_program_field: One of "Computer Engineering", "Electrical Engineering", or "Mechanical Engineering" corresponding to the chosen program.
    10) credit_hours_min: The stated minimum bachelor's credit requirement (e.g., "120").
    11) engineering_urls: Array of URLs verifying the engineering college name, ABET accreditation, and/or 120-credit minimum (program page, ABET page, catalog).
    12) residence_urls: Array of URLs verifying on-campus residence halls/dorms for undergraduates.
    13) dining_urls: Array of URLs verifying campus dining facilities with meal plans.
    14) recreation_urls: Array of URLs verifying a campus recreation/fitness facility.
    15) library_urls: Array of URLs verifying a main library or library system on campus.
    16) infrastructure_urls: Any additional URLs used to verify campus infrastructure items.
    17) international_office_name: Name of the International Student Services office (or equivalent) as stated.
    18) international_office_desc: A brief description of that office as stated (one sentence is enough).
    19) toefl_min: The minimum TOEFL iBT score for international undergraduate admissions as stated (string).
    20) ielts_min: The minimum IELTS score for international undergraduate admissions as stated (string).
    21) international_urls: Array of URLs verifying the international office and English proficiency requirements.

    IMPORTANT:
    - Extract ONLY from the provided answer; do not invent or infer.
    - For URLs, extract the actual links mentioned (plain or in markdown). Include full URLs with protocol.
    - If more than three universities are provided, keep ONLY the first three mentioned in the answer.
    - If fewer than three are provided, return however many are present.
    """


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def _combine_urls(*url_lists: Optional[List[str]]) -> List[str]:
    """Combine and de-duplicate URLs preserving order."""
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not u or not isinstance(u, str):
                continue
            u2 = u.strip()
            if not u2:
                continue
            if u2 not in seen:
                seen.add(u2)
                combined.append(u2)
    return combined


def _safe_name(u: UniversityItem, idx: int) -> str:
    return u.name if (u and u.name) else f"University #{idx + 1}"


# --------------------------------------------------------------------------- #
# Verification for a single university                                        #
# --------------------------------------------------------------------------- #
async def verify_single_university(evaluator: Evaluator, parent_node, uni: UniversityItem, idx: int) -> None:
    """
    Build the rubric subtree and run verifications for one university (idx in {0,1,2}).
    """
    uni_human_idx = idx + 1
    uni_node = evaluator.add_parallel(
        id=f"University_{uni_human_idx}",
        desc=f"{['First','Second','Third'][idx] if idx < 3 else f'#{uni_human_idx}th'} identified university meets all requirements",
        parent=parent_node,
        critical=False  # allow partial credit per university
    )

    # ---------------------- Institution Identity ---------------------- #
    identity_node = evaluator.add_parallel(
        id=f"U{uni_human_idx}_Institution_Identity",
        desc="Conference membership, founding date, and public status verification",
        parent=uni_node,
        critical=True
    )

    # Existence of at least one identity URL (gates other checks)
    identity_urls_exist = evaluator.add_custom_node(
        result=bool(uni.identity_urls),
        id=f"U{uni_human_idx}_Institution_URL",
        desc="Provides reference URL confirming institutional identity and conference membership",
        parent=identity_node,
        critical=True
    )

    # Official Name
    name_leaf = evaluator.add_leaf(
        id=f"U{uni_human_idx}_Official_Name",
        desc="Provides the official name of the university",
        parent=identity_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official name of the university is '{uni.name}'.",
        node=name_leaf,
        sources=uni.identity_urls,
        additional_instruction="Validate the official name from an authoritative page (university site, .edu domain, or Big Ten/credible source). Minor formatting or casing variations are acceptable."
    )

    # Big Ten membership (as of 2026)
    bigten_leaf = evaluator.add_leaf(
        id=f"U{uni_human_idx}_Big_Ten_Member",
        desc="Verifies current Big Ten Conference membership as of 2026",
        parent=identity_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_safe_name(uni, idx)} is a current member of the Big Ten Conference as of 2026.",
        node=bigten_leaf,
        sources=uni.identity_urls,
        additional_instruction="Confirm using the Big Ten official website or the university's official athletics/conference page that they are a Big Ten member in 2026."
    )

    # Public institution
    public_leaf = evaluator.add_leaf(
        id=f"U{uni_human_idx}_Public_Institution",
        desc="Confirms the university is a public institution, not private",
        parent=identity_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_safe_name(uni, idx)} is a public institution.",
        node=public_leaf,
        sources=uni.identity_urls,
        additional_instruction="Look for phrases like 'public university' or 'public research university' on official or authoritative pages."
    )

    # Founded year in required range
    founded_leaf = evaluator.add_leaf(
        id=f"U{uni_human_idx}_Founded_1848_1870",
        desc="Verifies founding year is between 1848 and 1870 inclusive",
        parent=identity_node,
        critical=True
    )
    founded_year_str = uni.founded_year if uni.founded_year else "UNKNOWN"
    await evaluator.verify(
        claim=f"{_safe_name(uni, idx)} was founded in {founded_year_str}, which is between 1848 and 1870 inclusive.",
        node=founded_leaf,
        sources=uni.identity_urls,
        additional_instruction="If multiple dates are shown (chartered vs classes began), use the 'founded/established' year. Verify that this year is within 1848–1870 inclusive."
    )

    # ---------------------- Enrollment ---------------------- #
    enroll_node = evaluator.add_parallel(
        id=f"U{uni_human_idx}_Enrollment",
        desc="Student enrollment verification",
        parent=uni_node,
        critical=True
    )

    # URLs for enrollment stats exist
    enroll_urls_exist = evaluator.add_custom_node(
        result=bool(uni.enrollment_urls),
        id=f"U{uni_human_idx}_Enrollment_URL",
        desc="Provides reference URL verifying enrollment statistics",
        parent=enroll_node,
        critical=True
    )

    # Total enrollment over 40k
    over40_leaf = evaluator.add_leaf(
        id=f"U{uni_human_idx}_Total_Enrollment_Over_40k",
        desc="Confirms total student enrollment exceeds 40,000",
        parent=enroll_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The current total student enrollment at {_safe_name(uni, idx)} exceeds 40,000 students.",
        node=over40_leaf,
        sources=uni.enrollment_urls,
        additional_instruction="Use official IR/Fact Book or enrollment summary pages. Accept statements like 'over 50,000' or exact totals > 40,000."
    )

    # Has undergraduate students
    ug_leaf = evaluator.add_leaf(
        id=f"U{uni_human_idx}_Has_Undergraduates",
        desc="Confirms the university enrolls undergraduate students",
        parent=enroll_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_safe_name(uni, idx)} enrolls undergraduate students.",
        node=ug_leaf,
        sources=uni.enrollment_urls,
        additional_instruction="Verify presence of undergraduate enrollment or undergraduate programs on official pages."
    )

    # ---------------------- Engineering Programs ---------------------- #
    eng_node = evaluator.add_parallel(
        id=f"U{uni_human_idx}_Engineering_Programs",
        desc="Engineering college and program accreditation verification",
        parent=uni_node,
        critical=True
    )

    eng_urls_exist = evaluator.add_custom_node(
        result=bool(uni.engineering_urls),
        id=f"U{uni_human_idx}_Engineering_URL",
        desc="Provides reference URL verifying engineering programs and ABET accreditation",
        parent=eng_node,
        critical=True
    )

    # Engineering College name
    eng_name_leaf = evaluator.add_leaf(
        id=f"U{uni_human_idx}_Engineering_College_Name",
        desc="Identifies the name of the College or School of Engineering",
        parent=eng_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The name of the engineering college at {_safe_name(uni, idx)} is '{uni.engineering_college_name}'.",
        node=eng_name_leaf,
        sources=uni.engineering_urls,
        additional_instruction="Check the engineering homepage or 'About' page for the formal name. Minor formatting/casing differences are acceptable."
    )

    # ABET-accredited program (one of CE/EE/ME)
    abet_leaf = evaluator.add_leaf(
        id=f"U{uni_human_idx}_ABET_Program",
        desc="Identifies at least one ABET-accredited program in Computer, Electrical, or Mechanical Engineering",
        parent=eng_node,
        critical=True
    )
    abet_prog_name = uni.abet_program_name or "the specified program"
    abet_prog_field = uni.abet_program_field or "Computer/Electrical/Mechanical Engineering"
    await evaluator.verify(
        claim=f"The undergraduate {abet_prog_name} program in the field of {abet_prog_field} at {_safe_name(uni, idx)} is ABET-accredited.",
        node=abet_leaf,
        sources=uni.engineering_urls,
        additional_instruction="Accept ABET's official listing or the college's accreditation page clearly stating ABET accreditation for CE/EE/ME."
    )

    # 120 credit minimum
    credit_leaf = evaluator.add_leaf(
        id=f"U{uni_human_idx}_120_Credit_Requirement",
        desc="Confirms bachelor's degree programs require minimum 120 credit hours",
        parent=eng_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Bachelor's degree programs at {_safe_name(uni, idx)} require at least 120 credit hours to graduate.",
        node=credit_leaf,
        sources=uni.engineering_urls,
        additional_instruction="Verify via undergraduate catalog, degree requirements, or program pages. Consider policy wordings like 'minimum 120 credit hours' as satisfying the requirement."
    )

    # ---------------------- Campus Infrastructure ---------------------- #
    infra_node = evaluator.add_parallel(
        id=f"U{uni_human_idx}_Campus_Infrastructure",
        desc="Campus facilities verification",
        parent=uni_node,
        critical=True
    )

    # Infrastructure URLs existence (can be any of the infra categories)
    infra_urls_any = _combine_urls(
        uni.residence_urls, uni.dining_urls, uni.recreation_urls, uni.library_urls, uni.infrastructure_urls
    )
    infra_urls_exist = evaluator.add_custom_node(
        result=bool(infra_urls_any),
        id=f"U{uni_human_idx}_Infrastructure_URL",
        desc="Provides reference URL(s) verifying campus infrastructure",
        parent=infra_node,
        critical=True
    )

    # Residence halls
    res_leaf = evaluator.add_leaf(
        id=f"U{uni_human_idx}_Residence_Halls",
        desc="Confirms existence of on-campus residence halls for undergraduates",
        parent=infra_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_safe_name(uni, idx)} has on-campus residence halls or dormitories available for undergraduate students.",
        node=res_leaf,
        sources=_combine_urls(uni.residence_urls, uni.infrastructure_urls),
        additional_instruction="Look for Housing/Residence Life pages that clearly indicate on-campus residence halls for undergraduates."
    )

    # Dining + meal plans
    dining_leaf = evaluator.add_leaf(
        id=f"U{uni_human_idx}_Dining_Facilities",
        desc="Confirms existence of campus dining facilities with meal plan options",
        parent=infra_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_safe_name(uni, idx)} has campus dining facilities that offer meal plan options for students.",
        node=dining_leaf,
        sources=_combine_urls(uni.dining_urls, uni.infrastructure_urls),
        additional_instruction="Look for 'Dining Services' or 'Meal Plans' pages indicating dining halls and meal plan options."
    )

    # Recreation/fitness facility
    rec_leaf = evaluator.add_leaf(
        id=f"U{uni_human_idx}_Recreation_Center",
        desc="Confirms existence of student recreation center or fitness facility",
        parent=infra_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_safe_name(uni, idx)} has a student recreation center or campus fitness facility.",
        node=rec_leaf,
        sources=_combine_urls(uni.recreation_urls, uni.infrastructure_urls),
        additional_instruction="Pages for 'Campus Recreation', 'Recreation Center', or 'Fitness Center' should satisfy this."
    )

    # Library system
    lib_leaf = evaluator.add_leaf(
        id=f"U{uni_human_idx}_Library_System",
        desc="Confirms existence of main library or library system on campus",
        parent=infra_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_safe_name(uni, idx)} has a main library or campus library system.",
        node=lib_leaf,
        sources=_combine_urls(uni.library_urls, uni.infrastructure_urls),
        additional_instruction="Use the university libraries' homepage or an 'About the Libraries' page."
    )

    # ---------------------- International Support ---------------------- #
    intl_node = evaluator.add_parallel(
        id=f"U{uni_human_idx}_International_Support",
        desc="International student services verification",
        parent=uni_node,
        critical=True
    )

    intl_urls_exist = evaluator.add_custom_node(
        result=bool(uni.international_urls),
        id=f"U{uni_human_idx}_International_URL",
        desc="Provides reference URL verifying international services and requirements",
        parent=intl_node,
        critical=True
    )

    # International office name
    intl_office_leaf = evaluator.add_leaf(
        id=f"U{uni_human_idx}_International_Office",
        desc="Identifies the International Student Services office or equivalent unit",
        parent=intl_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The international student services office at {_safe_name(uni, idx)} is called '{uni.international_office_name}'.",
        node=intl_office_leaf,
        sources=uni.international_urls,
        additional_instruction="Confirm the official office/unit name from the university website (e.g., 'International Student and Scholar Services' or similar)."
    )

    # English proficiency (TOEFL/IELTS)
    eng_prof_leaf = evaluator.add_leaf(
        id=f"U{uni_human_idx}_English_Proficiency",
        desc="Provides minimum TOEFL and/or IELTS score requirements for international undergraduates",
        parent=intl_node,
        critical=True
    )
    # Build a flexible claim depending on what was extracted
    if uni.toefl_min and uni.ielts_min:
        eng_claim = f"For international undergraduate admission at {_safe_name(uni, idx)}, the minimum required English proficiency scores are TOEFL iBT {uni.toefl_min} or IELTS {uni.ielts_min}."
    elif uni.toefl_min:
        eng_claim = f"For international undergraduate admission at {_safe_name(uni, idx)}, the minimum required TOEFL iBT score is {uni.toefl_min}."
    elif uni.ielts_min:
        eng_claim = f"For international undergraduate admission at {_safe_name(uni, idx)}, the minimum required IELTS score is {uni.ielts_min}."
    else:
        eng_claim = f"The official page for {_safe_name(uni, idx)} states minimum English proficiency requirements (TOEFL and/or IELTS) for international undergraduate admission."

    await evaluator.verify(
        claim=eng_claim,
        node=eng_prof_leaf,
        sources=uni.international_urls,
        additional_instruction="Verify minimum undergraduate entry scores. Accept pages that show test minima with 'or higher' language. Focus on undergraduate—not graduate—requirements."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Entry point for evaluating an answer for the Big Ten public universities (1848–1870) task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # universities evaluated independently
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

    # IMPORTANT: Make root non-critical to allow partial credit across universities
    root.critical = False
    root.desc = "Identifies up to three public Big Ten universities founded 1848–1870 with required evidence"

    # 1) Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # 2) Normalize to exactly three slots (pad with empty entries if needed)
    universities = list(extracted.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityItem())

    # 3) Build tree and verify for each university
    for i in range(3):
        await verify_single_university(evaluator, root, universities[i], i)

    # 4) Return the final structured evaluation summary
    return evaluator.get_summary()