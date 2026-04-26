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
TASK_ID = "academic_leaders_career_progression"
TASK_DESCRIPTION = """
Identify three academic leaders who meet all of the following career progression criteria:

1. Educational Background: Each individual must hold a doctoral degree (PhD or equivalent terminal degree) in natural sciences, engineering, mathematics, or a related STEM field from a recognized institution.

2. Early Career: Each individual must have completed a postdoctoral fellowship, research fellowship, or equivalent research position at a university.

3. Faculty Career: Each individual must have:
   - Served as a faculty member (assistant professor or higher rank) at a university
   - Held a prestigious research chair, named professorship, endowed chair, or received equivalent research recognition during their faculty career

4. Administrative Leadership: Each individual must have:
   - Served as a dean of a college, faculty, or school at a university for at least 3 years
   - Currently serve or have served (within 2023-2026) as a provost or chief academic officer at a university

5. Institutional Diversity: Each individual must have held positions at least 3 different universities during their academic career.

6. Career Timeline: Each individual's career progression from their first faculty appointment to their provost position must span at least 15 years.

For each of the three individuals, provide:
- Their full name
- A brief description of their career progression, including: doctoral institution and field, postdoctoral position, faculty positions (with any research chairs/distinctions), dean position (with institution and approximate duration), provost position (with institution and dates), and the universities where they held positions
- URL references that verify each aspect of their career (educational background, postdoctoral position, faculty career and research distinctions, dean position, provost position, and multi-institutional experience)
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Position(BaseModel):
    institution: Optional[str] = None
    title: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None


class Education(BaseModel):
    degree: Optional[str] = None
    field: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Postdoc(BaseModel):
    institution: Optional[str] = None
    title: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Faculty(BaseModel):
    positions: List[Position] = Field(default_factory=list)
    research_distinctions: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class Dean(BaseModel):
    institution: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Provost(BaseModel):
    institution: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PersonCareer(BaseModel):
    name: Optional[str] = None
    education: Education = Field(default_factory=Education)
    postdoc: Postdoc = Field(default_factory=Postdoc)
    faculty: Faculty = Field(default_factory=Faculty)
    dean: Dean = Field(default_factory=Dean)
    provost: Provost = Field(default_factory=Provost)
    all_institutions: List[str] = Field(default_factory=list)


class PeopleExtraction(BaseModel):
    persons: List[PersonCareer] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_people() -> str:
    return """
    Extract up to three (3) academic leaders described in the answer, with structured career data for each.
    For each person, extract the following fields exactly as presented in the answer:

    - name: Full name of the individual.

    - education:
        - degree: The doctoral degree type (e.g., PhD, DPhil, ScD, EngD, MD/PhD, etc.).
        - field: The doctoral degree field or discipline (e.g., Physics, Computer Science, Chemistry, Electrical Engineering).
        - institution: The awarding university.
        - year: The year the doctoral degree was awarded (if present).
        - urls: A list of URLs that verify this educational background. Only include URLs explicitly shown in the answer.

    - postdoc:
        - institution: The host university of the postdoctoral or equivalent research fellowship.
        - title: The title (e.g., Postdoctoral Fellow, Research Fellow).
        - start_year: The start year (if given; otherwise use null).
        - end_year: The end year or 'present' (if given; otherwise use null).
        - urls: A list of URLs that verify this postdoctoral/early research position.

    - faculty:
        - positions: An array where each element includes:
            - title: The academic rank/title (e.g., Assistant Professor, Associate Professor, Professor).
            - institution: The university of the appointment.
            - start_year: The start year (if stated).
            - end_year: The end year or 'present' (if stated).
        - research_distinctions: An array of named chairs, endowed chairs, distinguished/named professorships, or equivalent prestigious research recognitions (if any).
        - urls: A list of URLs that verify faculty appointments and/or research distinctions.

    - dean:
        - institution: The university where the person served as dean of a college/school/faculty.
        - start_year: The start year (if stated).
        - end_year: The end year or 'present' (if stated).
        - urls: A list of URLs that verify the dean position and (ideally) show dates.

    - provost:
        - institution: The university where the person serves/served as provost or chief academic officer.
        - start_year: The start year (if stated).
        - end_year: The end year or 'present' (if stated).
        - urls: A list of URLs that verify the provost/CAO position and (ideally) show dates.

    - all_institutions: A de-duplicated list of distinct universities across the person's education, postdoc, faculty, dean, and provost roles.

    RULES:
    - Return years as 4-digit strings when possible (e.g., "2011"); otherwise use the original text if not clearly a year.
    - Return URLs exactly as they appear in the answer (plain or markdown). Do not invent URLs.
    - If a field is not present in the answer, set it to null or an empty list as appropriate.
    - Make sure 'all_institutions' includes unique university names only (no departments/centers).
    - Return a JSON object { "persons": [ ... up to 3 persons ... ] }.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _clean_list_str(items: List[Optional[str]]) -> List[str]:
    return [x.strip() for x in items if isinstance(x, str) and x.strip()]


def _unique_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    out = []
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    seen.add(uu)
                    out.append(uu)
    return out


def _year_to_int(y: Optional[str]) -> Optional[int]:
    if not y or not isinstance(y, str):
        return None
    m = re.search(r"(19|20)\d{2}", y)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _earliest_faculty_start(person: PersonCareer) -> Optional[int]:
    years = []
    for pos in person.faculty.positions or []:
        yi = _year_to_int(pos.start_year)
        if yi:
            years.append(yi)
    return min(years) if years else None


def _provost_start(person: PersonCareer) -> Optional[int]:
    return _year_to_int(person.provost.start_year)


def _admin_urls(person: PersonCareer) -> List[str]:
    return _unique_urls(person.dean.urls, person.provost.urls)


def _all_evidence_urls(person: PersonCareer) -> List[str]:
    return _unique_urls(
        person.education.urls,
        person.postdoc.urls,
        person.faculty.urls,
        person.dean.urls,
        person.provost.urls,
    )


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_educational_background(evaluator: Evaluator, parent, p: PersonCareer, idx: int):
    node = evaluator.add_sequential(
        id=f"person_{idx}_educational_background",
        desc=f"Educational credentials verification for Person {idx}",
        parent=parent,
        critical=True
    )

    edu = p.education or Education()
    edu_urls = edu.urls or []

    # 1) Doctoral degree existence (terminal degree)
    leaf_degree = evaluator.add_leaf(
        id=f"person_{idx}_doctoral_degree",
        desc=f"Person {idx} holds a doctoral/terminal degree from a recognized institution",
        parent=node,
        critical=True
    )
    degree = edu.degree or ""
    inst = edu.institution or ""
    claim_degree = (
        f"{p.name or 'This person'} holds an earned doctoral or equivalent terminal degree "
        f"(e.g., PhD/DPhil/ScD/EngD) from {inst}. "
        f"Stated degree: '{degree}'."
    )
    await evaluator.verify(
        claim=claim_degree,
        node=leaf_degree,
        sources=edu_urls,
        additional_instruction="Confirm that the person indeed earned a doctoral or equivalent terminal degree from a recognized university, as stated."
    )

    # 2) STEM field check
    leaf_stem = evaluator.add_leaf(
        id=f"person_{idx}_stem_field",
        desc=f"Person {idx}'s doctoral field is in STEM",
        parent=node,
        critical=True
    )
    field = edu.field or ""
    claim_stem = (
        f"The person's doctoral field '{field}' is a STEM field "
        f"(natural sciences, engineering, mathematics, or a closely related STEM area)."
    )
    await evaluator.verify(
        claim=claim_stem,
        node=leaf_stem,
        sources=edu_urls,
        additional_instruction="Classify the doctoral field as STEM if it clearly falls within natural sciences, engineering, mathematics, computer science, or closely related technical disciplines. If ambiguous or clearly non-STEM (e.g., business, education policy), mark as incorrect."
    )

    # 3) Evidence URLs present
    evaluator.add_custom_node(
        result=bool(_unique_urls(edu_urls)),
        id=f"person_{idx}_education_evidence",
        desc=f"URL reference provided for Person {idx}'s educational background",
        parent=node,
        critical=True
    )


async def verify_early_career(evaluator: Evaluator, parent, p: PersonCareer, idx: int):
    node = evaluator.add_parallel(
        id=f"person_{idx}_early_career",
        desc=f"Early career positions for Person {idx}",
        parent=parent,
        critical=True
    )

    # Evidence presence first (as a gating critical sibling)
    evaluator.add_custom_node(
        result=bool(_unique_urls(p.postdoc.urls)),
        id=f"person_{idx}_postdoc_evidence",
        desc=f"URL reference provided for Person {idx}'s postdoctoral or early research position",
        parent=node,
        critical=True
    )

    leaf_postdoc = evaluator.add_leaf(
        id=f"person_{idx}_postdoctoral_position",
        desc=f"Person {idx} completed a postdoctoral/research fellowship at a university",
        parent=node,
        critical=True
    )
    postdoc_inst = p.postdoc.institution or ""
    postdoc_title = p.postdoc.title or "postdoctoral/research fellowship"
    claim_postdoc = (
        f"{p.name or 'This person'} completed a {postdoc_title} at {postdoc_inst}, "
        f"which is a university-based postdoctoral or equivalent research position."
    )
    await evaluator.verify(
        claim=claim_postdoc,
        node=leaf_postdoc,
        sources=p.postdoc.urls or [],
        additional_instruction="Verify that this was a postdoctoral/research fellowship (or clearly equivalent early-career research position) hosted by a university."
    )


async def verify_faculty_career(evaluator: Evaluator, parent, p: PersonCareer, idx: int):
    node = evaluator.add_parallel(
        id=f"person_{idx}_faculty_career",
        desc=f"Faculty career progression for Person {idx}",
        parent=parent,
        critical=True
    )

    # Evidence presence first
    evaluator.add_custom_node(
        result=bool(_unique_urls(p.faculty.urls)),
        id=f"person_{idx}_faculty_evidence",
        desc=f"URL reference provided for Person {idx}'s faculty positions and research distinctions",
        parent=node,
        critical=True
    )

    # Faculty appointment
    leaf_faculty = evaluator.add_leaf(
        id=f"person_{idx}_faculty_appointment",
        desc=f"Person {idx} served as a faculty member (assistant professor or higher)",
        parent=node,
        critical=True
    )
    pos_sample = p.faculty.positions[0] if (p.faculty.positions or []) else Position()
    pos_title = pos_sample.title or "faculty position"
    pos_inst = pos_sample.institution or (p.faculty.positions[0].institution if p.faculty.positions else "")
    claim_faculty = (
        f"{p.name or 'This person'} served as a faculty member (assistant professor or higher) at a university. "
        f"Example appointment: '{pos_title}' at '{pos_inst}'."
    )
    await evaluator.verify(
        claim=claim_faculty,
        node=leaf_faculty,
        sources=p.faculty.urls or [],
        additional_instruction="Confirm that at least one listed appointment is a faculty role at the rank of Assistant Professor, Associate Professor, Professor, or equivalent."
    )

    # Research distinction (named/endowed chair, etc.)
    leaf_dist = evaluator.add_leaf(
        id=f"person_{idx}_research_distinction",
        desc=f"Person {idx} held a prestigious research chair or named/endowed professorship",
        parent=node,
        critical=True
    )
    distinctions = _clean_list_str(p.faculty.research_distinctions or [])
    dist_text = "; ".join(distinctions) if distinctions else "no named distinction provided"
    claim_dist = (
        f"{p.name or 'This person'} held a prestigious research chair, named professorship, endowed chair, "
        f"or equivalent research recognition during their faculty career. Noted distinctions: {dist_text}."
    )
    await evaluator.verify(
        claim=claim_dist,
        node=leaf_dist,
        sources=p.faculty.urls or [],
        additional_instruction="Look for explicit phrases such as 'Endowed Chair', 'Named Professor', 'Distinguished Professor', or similar prestigious research titles on the cited pages."
    )


async def verify_administrative_progression(evaluator: Evaluator, parent, p: PersonCareer, idx: int):
    node = evaluator.add_parallel(
        id=f"person_{idx}_administrative_progression",
        desc=f"Administrative leadership progression for Person {idx}",
        parent=parent,
        critical=True
    )

    # Evidence presence first
    evaluator.add_custom_node(
        result=bool(_admin_urls(p)),
        id=f"person_{idx}_admin_evidence",
        desc=f"URL references provided for Person {idx}'s dean and provost positions with durations",
        parent=node,
        critical=True
    )

    # Dean position (>= 3 years)
    leaf_dean = evaluator.add_leaf(
        id=f"person_{idx}_dean_position",
        desc=f"Person {idx} served as a dean for at least 3 years",
        parent=node,
        critical=True
    )
    dean_inst = p.dean.institution or ""
    d_start = p.dean.start_year or ""
    d_end = p.dean.end_year or "present"
    claim_dean = (
        f"{p.name or 'This person'} served as a dean (of a college/school/faculty) at {dean_inst} "
        f"from {d_start} to {d_end}, and the tenure lasted at least 3 years."
    )
    await evaluator.verify(
        claim=claim_dean,
        node=leaf_dean,
        sources=_admin_urls(p),
        additional_instruction="Use the cited pages to check the dean appointment dates. If end year is 'present', estimate duration as present year minus start year. Confirm the total is 3 or more years."
    )

    # Provost within 2023–2026
    leaf_provost = evaluator.add_leaf(
        id=f"person_{idx}_provost_position",
        desc=f"Person {idx} served as a provost (or CAO) within 2023–2026",
        parent=node,
        critical=True
    )
    prov_inst = p.provost.institution or ""
    p_start = p.provost.start_year or ""
    p_end = p.provost.end_year or "present"
    claim_provost = (
        f"{p.name or 'This person'} served as a provost or chief academic officer at {prov_inst}, "
        f"with service occurring at some point during 2023–2026 inclusive. "
        f"Stated dates: {p_start} to {p_end}."
    )
    await evaluator.verify(
        claim=claim_provost,
        node=leaf_provost,
        sources=p.provost.urls or [],
        additional_instruction="Confirm that their provost (or CAO) service overlaps any time in 2023, 2024, 2025, or 2026."
    )


async def verify_institutional_diversity(evaluator: Evaluator, parent, p: PersonCareer, idx: int):
    # IMPORTANT: To allow a non-critical child (geographic_diversity), the parent must be non-critical
    node = evaluator.add_parallel(
        id=f"person_{idx}_institutional_diversity",
        desc=f"Multi-institutional experience for Person {idx}",
        parent=parent,
        critical=False  # Adjusted to allow a non-critical child per framework constraint
    )

    # Evidence presence first (critical)
    evaluator.add_custom_node(
        result=bool(_all_evidence_urls(p)),
        id=f"person_{idx}_institution_evidence",
        desc=f"URL references provided documenting Person {idx}'s positions at multiple institutions",
        parent=node,
        critical=True
    )

    # At least 3 different universities (critical)
    leaf_multi = evaluator.add_leaf(
        id=f"person_{idx}_multiple_institutions",
        desc=f"Person {idx} held positions at least 3 different universities",
        parent=node,
        critical=True
    )
    insts = _clean_list_str(p.all_institutions or [])
    listed = "; ".join(insts) if insts else "none listed"
    claim_multi = (
        f"{p.name or 'This person'} held positions at least three different universities. "
        f"Claimed institutions: {listed}."
    )
    await evaluator.verify(
        claim=claim_multi,
        node=leaf_multi,
        sources=_all_evidence_urls(p),
        additional_instruction="Verify that there are at least three distinct universities. Treat multiple campuses of one system as distinct if they are recognized as separate universities (e.g., UC Berkeley vs. UC Davis). Ignore internal departments/centers."
    )

    # Geographic diversity (non-critical)
    leaf_geo = evaluator.add_leaf(
        id=f"person_{idx}_geographic_diversity",
        desc=f"Person {idx}'s institutions span >=2 countries OR >=3 different U.S. states",
        parent=node,
        critical=False
    )
    claim_geo = (
        f"The listed institutions for {p.name or 'this person'} are located in at least two different countries "
        f"OR in at least three different U.S. states."
    )
    await evaluator.verify(
        claim=claim_geo,
        node=leaf_geo,
        sources=_all_evidence_urls(p),
        additional_instruction="Check university locations across the cited pages. If locations are not explicit, rely on institution names to infer locations when reasonable."
    )


async def verify_career_timeline(evaluator: Evaluator, parent, p: PersonCareer, idx: int):
    node = evaluator.add_parallel(
        id=f"person_{idx}_career_timeline",
        desc=f"Career progression timeline for Person {idx}",
        parent=parent,
        critical=True
    )

    # Evidence presence first (critical)
    evaluator.add_custom_node(
        result=bool(_all_evidence_urls(p)),
        id=f"person_{idx}_timeline_evidence",
        desc=f"URL references provided with dates documenting Person {idx}'s career timeline",
        parent=node,
        critical=True
    )

    # Duration >= 15 years from first faculty to provost
    leaf_dur = evaluator.add_leaf(
        id=f"person_{idx}_progression_duration",
        desc=f"Person {idx}'s progression from first faculty to provost spans >= 15 years",
        parent=node,
        critical=True
    )
    first_faculty_year = _earliest_faculty_start(p)
    provost_year = _provost_start(p)
    fx = str(first_faculty_year) if first_faculty_year else (p.faculty.positions[0].start_year if (p.faculty.positions) else "unknown")
    py = str(provost_year) if provost_year else (p.provost.start_year or "unknown")

    claim_dur = (
        f"For {p.name or 'this person'}, the time from first faculty appointment year '{fx}' "
        f"to provost appointment year '{py}' is at least 15 years."
    )
    await evaluator.verify(
        claim=claim_dur,
        node=leaf_dur,
        sources=_all_evidence_urls(p),
        additional_instruction="Use dates on the cited pages to determine the first faculty appointment year and the provost appointment (or service) year. Compute the difference; it must be >= 15."
    )


async def verify_one_person(evaluator: Evaluator, root, person: PersonCareer, index_one_based: int):
    person_node = evaluator.add_parallel(
        id=f"person_{index_one_based}",
        desc=f"{['First','Second','Third'][min(index_one_based-1,2)]} academic leader identification and verification",
        parent=root,
        critical=False
    )

    # Run sub-verifications
    await verify_educational_background(evaluator, person_node, person, index_one_based)
    await verify_early_career(evaluator, person_node, person, index_one_based)
    await verify_faculty_career(evaluator, person_node, person, index_one_based)
    await verify_administrative_progression(evaluator, person_node, person, index_one_based)
    await verify_institutional_diversity(evaluator, person_node, person, index_one_based)
    await verify_career_timeline(evaluator, person_node, person, index_one_based)


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

    # Extract people data
    people = await evaluator.extract(
        prompt=prompt_extract_people(),
        template_class=PeopleExtraction,
        extraction_name="people_extraction"
    )

    # Normalize to exactly 3 persons (pad with empty if needed)
    persons: List[PersonCareer] = list(people.persons or [])
    persons = persons[:3]
    while len(persons) < 3:
        persons.append(PersonCareer())

    # Optional: record criteria as GT/context
    evaluator.add_ground_truth({
        "required_individuals": 3,
        "criteria": [
            "Doctoral degree in STEM",
            "Completed postdoc/research fellowship at a university",
            "Faculty member (assistant professor or higher)",
            "Held prestigious named/endowed chair or equivalent",
            "Dean for >= 3 years",
            "Provost (or CAO) within 2023–2026",
            "Held positions at >= 3 universities",
            ">= 15 years from first faculty to provost"
        ]
    }, gt_type="task_criteria")

    # Verify each person
    for i, person in enumerate(persons, start=1):
        await verify_one_person(evaluator, root, person, i)

    # Return the final evaluation summary
    return evaluator.get_summary()