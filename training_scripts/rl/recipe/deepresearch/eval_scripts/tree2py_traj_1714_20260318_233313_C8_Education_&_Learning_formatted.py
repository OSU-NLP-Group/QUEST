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
TASK_ID = "upper_midwest_d1_hockey_teacher_ed"
TASK_DESCRIPTION = """
A high school student from Minnesota is planning to pursue both competitive ice hockey and a career in education. They are looking for public universities in the Upper Midwest region where they can play NCAA Division I men's ice hockey at an established program, earn a degree in teacher education from an accredited program, benefit from regional tuition agreements, and access comprehensive academic support and enrichment opportunities.

Identify three public universities that meet ALL of the following requirements:

Athletic Requirements:
1. The university must have an NCAA Division I men's ice hockey program
2. The hockey program must compete in either the Big Ten Conference or the National Collegiate Hockey Conference (NCHC)
3. The program must have been competing at the Division I level for at least 10 years (not a recently transitioned program)

Geographic and Institutional Requirements:
4. The university must be a public institution located in Minnesota, Wisconsin, Michigan, or North Dakota

Academic Program Requirements:
5. The university must offer undergraduate teacher education or educator preparation programs leading to teaching licensure
6. The teacher education program must be accredited by the Council for the Accreditation of Educator Preparation (CAEP)

Institutional Characteristics:
7. The university's total student enrollment (undergraduate and graduate combined) must be between 9,000 and 55,000 students
8. The university must have an established honors college or honors program available to undergraduate students
9. The university must have clearly published admission requirements, including minimum GPA expectations or test score ranges for incoming freshmen

Additional Academic Opportunities:
10. The university must offer study abroad programs or international education opportunities for undergraduate students
11. The university must provide undergraduate research opportunities that allow students to engage in faculty-mentored research

Financial Considerations:
12. The university must participate in at least one tuition reciprocity agreement that benefits students from neighboring Upper Midwest states (such as Minnesota-Wisconsin reciprocity, Minnesota-North Dakota reciprocity, or the Midwest Student Exchange Program)

For each university you identify, provide:
- The full official name of the university
- Its location (city and state)
- The specific hockey conference in which it competes (Big Ten or NCHC)
- A reference URL from the university's official website confirming its NCAA Division I hockey program
- A reference URL confirming CAEP accreditation of its teacher education program
- A reference URL confirming participation in a tuition reciprocity agreement
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    # Core identity
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    # Hockey
    hockey_conference: Optional[str] = None  # Expect values like "Big Ten", "B1G", "NCHC"
    hockey_program_url: Optional[str] = None  # Prefer official athletics or team page
    program_history_url: Optional[str] = None  # Optional: specific team history/records page for longevity

    # Academics
    teacher_ed_url: Optional[str] = None  # Undergrad teacher education/licensure program page
    caep_accreditation_url: Optional[str] = None  # CAEP accreditation confirmation page (CAEP site or university page)

    # Institutional
    enrollment_info_url: Optional[str] = None  # Facts/enrollment dashboard page
    enrollment_text: Optional[str] = None  # Any enrollment figure or statement extracted from the answer
    public_status_url: Optional[str] = None  # Page that makes clear the institution is public

    # Student experience
    honors_url: Optional[str] = None
    admissions_requirements_url: Optional[str] = None  # Freshman requirements page (GPA/test info)
    study_abroad_url: Optional[str] = None
    undergraduate_research_url: Optional[str] = None

    # Financial reciprocity
    tuition_reciprocity_url: Optional[str] = None  # Official page confirming participation in reciprocity or MSEP

    # Catch-all extra URLs that the answer may provide (conference/athletics/college pages, etc.)
    additional_urls: List[str] = Field(default_factory=list)


class UniversityListExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to the first 5 universities explicitly mentioned in the answer that are proposed as options for the student.
    For each university, extract the following fields exactly as presented (use null if not present). Use the most authoritative URLs when multiple are shown:
    - name: Full official name of the university
    - city: City where the main campus is located
    - state: Two-letter or full state name (MN / WI / MI / ND or full names)
    - hockey_conference: The men's hockey conference stated (e.g., "Big Ten", "B1G", "NCHC")
    - hockey_program_url: A URL (preferably an official university athletics/team page) that confirms NCAA Division I men's ice hockey at that university
    - program_history_url: A URL (preferably official athletics or a reputable archival page) that supports that the hockey program has been NCAA Division I for at least 10 years
    - teacher_ed_url: A university page describing undergraduate teacher education / educator preparation / teacher licensure programs
    - caep_accreditation_url: A URL that confirms CAEP accreditation (acceptable: CAEP website or an official university accreditation page mentioning CAEP)
    - enrollment_info_url: A university page that states total student enrollment (combined undergraduate + graduate)
    - enrollment_text: The enrollment figure or description as stated in the answer (e.g., "about 31,000 total students"); do not invent numbers
    - public_status_url: A university facts/about page that shows the institution is public (if provided)
    - honors_url: A page for an established Honors College or Honors Program for undergraduates
    - admissions_requirements_url: A freshman/first-year admissions requirements or standards page indicating GPA expectations and/or test score ranges/policy
    - study_abroad_url: A page describing undergraduate study abroad or international education programs
    - undergraduate_research_url: A page describing structured undergraduate research opportunities with faculty mentorship
    - tuition_reciprocity_url: A page confirming tuition reciprocity that benefits Upper Midwest students (e.g., MN–WI reciprocity, MN–ND reciprocity, MSEP)
    - additional_urls: Any other URLs mentioned that relate to these checks (conference pages, athletics articles, college accreditation pages, etc.)
    
    Return a JSON object with a 'universities' array where each element has these fields.
    If a field is not present in the answer, set it to null (or [] for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(urls: List[Optional[str]]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and u.strip()]

def _gather_all_urls(u: UniversityItem) -> List[str]:
    bundle = _non_empty([
        u.hockey_program_url,
        u.program_history_url,
        u.teacher_ed_url,
        u.caep_accreditation_url,
        u.enrollment_info_url,
        u.public_status_url,
        u.honors_url,
        u.admissions_requirements_url,
        u.study_abroad_url,
        u.undergraduate_research_url,
        u.tuition_reciprocity_url,
    ])
    # Extend with additional_urls safely
    if u.additional_urls:
        bundle.extend([x for x in u.additional_urls if isinstance(x, str) and x.strip()])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for x in bundle:
        if x not in seen:
            deduped.append(x)
            seen.add(x)
    return deduped

def _ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n] if 0 <= n < 5 else f"#{n+1}"


# --------------------------------------------------------------------------- #
# Verification logic for a single university                                  #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    uni_index: int,
) -> None:
    """
    Build the verification subtree and run verifications for a single university.
    This follows the rubric's per-university criteria as critical leaf checks.
    """
    # University container node (parallel, non-critical to allow partial credit across universities)
    uni_node = evaluator.add_parallel(
        id=f"University_{uni_index}",
        desc=f"{_ordinal(uni_index-1)} university identified meets all requirements",
        parent=parent_node,
        critical=False
    )

    # Convenient overall URL pool (used as fallback to avoid non-evidenced simple checks)
    all_urls = _gather_all_urls(uni)

    # 0) Pre-checks for required reference URLs explicitly requested by the task (critical)
    #    These are separate from rubric leaves but enforce the "provide a reference URL" requirement:
    evaluator.add_custom_node(
        result=bool(uni.hockey_program_url and uni.hockey_program_url.strip()),
        id=f"u{uni_index}_hockey_url_provided",
        desc="Provided a reference URL from the university's official website confirming NCAA Division I men's ice hockey",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.caep_accreditation_url and uni.caep_accreditation_url.strip()),
        id=f"u{uni_index}_caep_url_provided",
        desc="Provided a reference URL confirming CAEP accreditation of the teacher education program",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.tuition_reciprocity_url and uni.tuition_reciprocity_url.strip()),
        id=f"u{uni_index}_reciprocity_url_provided",
        desc="Provided a reference URL confirming participation in a tuition reciprocity agreement",
        parent=uni_node,
        critical=True
    )

    # 1) Location_and_Type (public institution in MN/WI/MI/ND)
    loc_node = evaluator.add_leaf(
        id=f"u{uni_index}_location_and_type",
        desc="University is a public institution in Minnesota, Wisconsin, Michigan, or North Dakota",
        parent=uni_node,
        critical=True
    )
    state_str = (uni.state or "").strip()
    city_str = (uni.city or "").strip()
    name_str = (uni.name or "the university").strip()
    allowed_states = {"MN", "Minnesota", "WI", "Wisconsin", "MI", "Michigan", "ND", "North Dakota"}
    # Prefer institutional pages for public status and location
    loc_sources = _non_empty([uni.public_status_url, uni.enrollment_info_url, uni.admissions_requirements_url]) or all_urls or None
    loc_claim = (
        f"{name_str} is a public institution located in {city_str}, {state_str}, and the state is one of MN/WI/MI/ND."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=loc_sources,
        additional_instruction="Verify both (a) that the institution is PUBLIC and (b) the campus location is in MN, WI, MI, or ND. "
                               "Accept state matches by abbreviation or full name. Use only the provided pages as evidence."
    )

    # 2) Division_I_Hockey
    d1_node = evaluator.add_leaf(
        id=f"u{uni_index}_division1_hockey",
        desc="University has an NCAA Division I men's ice hockey program",
        parent=uni_node,
        critical=True
    )
    d1_sources = _non_empty([uni.hockey_program_url]) or all_urls or None
    d1_claim = "This university sponsors an NCAA Division I men's ice hockey program (varsity, not club/ACHA)."
    await evaluator.verify(
        claim=d1_claim,
        node=d1_node,
        sources=d1_sources,
        additional_instruction="Only count varsity NCAA Division I men's ice hockey. If the page indicates club/ACHA or women's only, it should fail."
    )

    # 3) Conference_Affiliation
    conf_node = evaluator.add_leaf(
        id=f"u{uni_index}_conference_affiliation",
        desc="Hockey program competes in Big Ten Conference or National Collegiate Hockey Conference (NCHC)",
        parent=uni_node,
        critical=True
    )
    conf_clean = (uni.hockey_conference or "").strip()
    # Accept synonyms automatically in instruction (e.g., 'B1G' <-> 'Big Ten')
    conf_sources = _non_empty([uni.hockey_program_url]) or all_urls or None
    if conf_clean:
        conf_claim = f"The men's hockey program competes in the {conf_clean} conference, which is either the Big Ten Conference or the National Collegiate Hockey Conference (NCHC)."
    else:
        conf_claim = "The men's hockey program competes in either the Big Ten Conference or the National Collegiate Hockey Conference (NCHC)."
    await evaluator.verify(
        claim=conf_claim,
        node=conf_node,
        sources=conf_sources,
        additional_instruction="Confirm the specific conference is Big Ten (allow 'B1G' as equivalent) or NCHC. "
                               "Do not count other conferences (e.g., CCHA, WCHA)."
    )

    # 4) Established_Program (>=10 years at D-I)
    estab_node = evaluator.add_leaf(
        id=f"u{uni_index}_established_program",
        desc="Hockey program has competed at Division I level for at least 10 years",
        parent=uni_node,
        critical=True
    )
    estab_sources = _non_empty([uni.program_history_url, uni.hockey_program_url]) or all_urls or None
    estab_claim = (
        "The men's hockey program has been competing at the NCAA Division I level for at least 10 years (i.e., established, not a recent transition)."
    )
    await evaluator.verify(
        claim=estab_claim,
        node=estab_node,
        sources=estab_sources,
        additional_instruction="Use the provided pages to infer longevity (e.g., founding year, years of D-I participation, long historical records). "
                               "Current date is March 2026; thus at least since the 2015–16 season or earlier."
    )

    # 5) Teacher_Education (undergrad + licensure)
    teach_node = evaluator.add_leaf(
        id=f"u{uni_index}_teacher_education",
        desc="University offers undergraduate teacher education or educator preparation programs",
        parent=uni_node,
        critical=True
    )
    teach_sources = _non_empty([uni.teacher_ed_url, uni.caep_accreditation_url]) or all_urls or None
    teach_claim = "The university offers undergraduate teacher education / educator preparation programs that lead to state teaching licensure."
    await evaluator.verify(
        claim=teach_claim,
        node=teach_node,
        sources=teach_sources,
        additional_instruction="Look for explicit mention of undergraduate teacher education and licensure pathways (elementary/secondary/etc.)."
    )

    # 6) CAEP_Accreditation
    caep_node = evaluator.add_leaf(
        id=f"u{uni_index}_caep_accreditation",
        desc="Teacher education program is accredited by CAEP",
        parent=uni_node,
        critical=True
    )
    caep_sources = _non_empty([uni.caep_accreditation_url]) or all_urls or None
    caep_claim = "The institution's teacher education program is accredited by the Council for the Accreditation of Educator Preparation (CAEP)."
    await evaluator.verify(
        claim=caep_claim,
        node=caep_node,
        sources=caep_sources,
        additional_instruction="Accept explicit CAEP accreditation statements on university pages or CAEP's official accredited program listings."
    )

    # 7) Enrollment_Size (9,000 to 55,000 total)
    enroll_node = evaluator.add_leaf(
        id=f"u{uni_index}_enrollment_size",
        desc="Total student enrollment is between 9,000 and 55,000 students",
        parent=uni_node,
        critical=True
    )
    enroll_sources = _non_empty([uni.enrollment_info_url]) or all_urls or None
    etext = (uni.enrollment_text or "").strip()
    enroll_claim = "The university's total student enrollment (undergraduate + graduate combined) is between 9,000 and 55,000 students."
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_node,
        sources=enroll_sources,
        additional_instruction="Prefer official facts/enrollment dashboards. If separate undergraduate and graduate counts are given, sum them. "
                               f"Any total that falls within [9,000, 55,000] should pass. Enrollment detail from answer: '{etext}'"
    )

    # 8) Honors_Program
    honors_node = evaluator.add_leaf(
        id=f"u{uni_index}_honors_program",
        desc="University has an established honors college or honors program for undergraduates",
        parent=uni_node,
        critical=True
    )
    honors_sources = _non_empty([uni.honors_url]) or all_urls or None
    honors_claim = "The university has an established undergraduate Honors College or Honors Program."
    await evaluator.verify(
        claim=honors_claim,
        node=honors_node,
        sources=honors_sources,
        additional_instruction="Look for official 'Honors Program' or 'Honors College' pages describing structured honors opportunities."
    )

    # 9) Tuition_Reciprocity
    recip_node = evaluator.add_leaf(
        id=f"u{uni_index}_tuition_reciprocity",
        desc="University participates in at least one tuition reciprocity agreement benefiting students from neighboring Upper Midwest states",
        parent=uni_node,
        critical=True
    )
    recip_sources = _non_empty([uni.tuition_reciprocity_url]) or all_urls or None
    recip_claim = "The university participates in a tuition reciprocity agreement that benefits Upper Midwest students (e.g., MN–WI reciprocity, MN–ND reciprocity, or MSEP)."
    await evaluator.verify(
        claim=recip_claim,
        node=recip_node,
        sources=recip_sources,
        additional_instruction="Confirm explicit participation in a reciprocity program (examples include MN–WI, MN–ND, or the Midwest Student Exchange Program)."
    )

    # 10) Study_Abroad
    abroad_node = evaluator.add_leaf(
        id=f"u{uni_index}_study_abroad",
        desc="University offers study abroad programs or international education opportunities",
        parent=uni_node,
        critical=True
    )
    abroad_sources = _non_empty([uni.study_abroad_url]) or all_urls or None
    abroad_claim = "The university offers study abroad or international education opportunities for undergraduate students."
    await evaluator.verify(
        claim=abroad_claim,
        node=abroad_node,
        sources=abroad_sources,
        additional_instruction="Look for pages by Study Abroad/International Programs/Global Education offices with undergraduate offerings."
    )

    # 11) Research_Opportunities
    research_node = evaluator.add_leaf(
        id=f"u{uni_index}_research_opportunities",
        desc="University provides undergraduate research opportunities with faculty mentorship",
        parent=uni_node,
        critical=True
    )
    research_sources = _non_empty([uni.undergraduate_research_url]) or all_urls or None
    research_claim = "The university offers structured undergraduate research opportunities that involve faculty mentorship."
    await evaluator.verify(
        claim=research_claim,
        node=research_node,
        sources=research_sources,
        additional_instruction="Accept programs like UROP, SURF, departmental research-for-credit, or similar formal undergraduate research initiatives."
    )

    # 12) Admission_Requirements
    admit_node = evaluator.add_leaf(
        id=f"u{uni_index}_admission_requirements",
        desc="University has clearly published admission requirements including GPA expectations or test score information",
        parent=uni_node,
        critical=True
    )
    admit_sources = _non_empty([uni.admissions_requirements_url]) or all_urls or None
    admit_claim = "The university clearly publishes first-year/freshman admission requirements that include either minimum GPA expectations and/or test score ranges/policy."
    await evaluator.verify(
        claim=admit_claim,
        node=admit_node,
        sources=admit_sources,
        additional_instruction="Accept publication of (a) minimum or recommended GPA values OR (b) test score ranges (e.g., middle 50%) or explicit testing policy details. "
                               "If the page lacks any GPA expectations and any test score ranges/policy, it should fail."
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
    Evaluate an answer for the Upper Midwest D-I men's hockey + teacher education university selection task.
    Returns a structured evaluation summary produced by the evaluator.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Universities are independent; allow partial credit across them
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

    # Extract proposed universities and their evidence URLs
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversityListExtraction,
        extraction_name="extracted_universities"
    )

    # Keep exactly 3 universities (pad with empty items if fewer)
    universities: List[UniversityItem] = list(extracted.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityItem())

    # Build Task_Completion node (parallel, non-critical to avoid consistency constraint and allow partial scoring)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Student identifies three public universities that meet all specified criteria for a prospective student-athlete interested in ice hockey and teacher education",
        parent=root,
        critical=False
    )

    # Verify each university
    for idx, uni in enumerate(universities, start=1):
        await verify_university(evaluator, task_node, uni, idx)

    # Return the final structured summary
    return evaluator.get_summary()