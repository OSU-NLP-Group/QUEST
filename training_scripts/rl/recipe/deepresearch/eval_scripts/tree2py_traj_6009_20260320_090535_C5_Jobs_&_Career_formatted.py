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
TASK_ID = "academic_leader_2022_2026_transition"
TASK_DESCRIPTION = (
    "Identify an academic leader who holds both a Juris Doctor (JD) degree and a Doctor of Philosophy (PhD) degree in a field outside of pure law "
    "(such as history, sociology, or science and technology studies). This person must have served as dean of a law school at a Research 1 (R1) "
    "university or equivalent major institution for at least 5 years. After their tenure as law school dean, they transitioned to become the chancellor "
    "or president of a different university—specifically, a flagship state university or R1 research institution—with the appointment being announced "
    "or becoming effective between January 1, 2022, and December 31, 2026 (inclusive). Additionally, this person must have held faculty positions at "
    "at least two different universities during their career, including experience as a law faculty member. Provide the person's full first and last name."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Degree(BaseModel):
    degree_type: Optional[str] = None  # e.g., "JD", "J.D.", "Juris Doctor", "PhD", "DPhil", "Doctor of Philosophy"
    field: Optional[str] = None        # e.g., "History", "Sociology", "Science and Technology Studies"
    institution: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DeanRole(BaseModel):
    law_school: Optional[str] = None            # e.g., "UCLA School of Law"
    university: Optional[str] = None            # Parent university, e.g., "University of California, Los Angeles"
    start_date: Optional[str] = None            # Free text date (e.g., "July 2015", "2015")
    end_date: Optional[str] = None              # Free text date (e.g., "June 2022", "2022")
    duration_years: Optional[str] = None        # e.g., "7", "at least 5", keep as string
    sources: List[str] = Field(default_factory=list)


class Appointment(BaseModel):
    title: Optional[str] = None                 # e.g., "Chancellor", "President"
    university: Optional[str] = None            # Destination institution name
    announcement_date: Optional[str] = None     # e.g., "May 16, 2022"
    effective_date: Optional[str] = None        # e.g., "July 1, 2022"
    sources: List[str] = Field(default_factory=list)


class FacultyAppointment(BaseModel):
    university: Optional[str] = None
    school_or_dept: Optional[str] = None        # e.g., "School of Law", "Department of Sociology"
    role_title: Optional[str] = None            # e.g., "Professor of Law", "Associate Professor"
    is_law_faculty: Optional[bool] = None       # True if explicitly a law faculty appointment
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CandidateExtraction(BaseModel):
    # Person
    full_name: Optional[str] = None

    # Degrees
    degrees: List[Degree] = Field(default_factory=list)

    # Law school dean experience
    law_dean: Optional[DeanRole] = None

    # Destination leadership appointment (president/chancellor)
    new_position: Optional[Appointment] = None

    # Source groups for institution classifications (if provided)
    law_school_r1_sources: List[str] = Field(default_factory=list)
    destination_inst_type_sources: List[str] = Field(default_factory=list)

    # Faculty background
    faculty_positions: List[FacultyAppointment] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_candidate() -> str:
    return """
    Extract structured information about the single person proposed in the answer who meets the task criteria.
    Return a JSON following this schema:

    {
      "full_name": string or null,
      "degrees": [
        {
          "degree_type": string or null,     // e.g., "JD", "J.D.", "Juris Doctor", "PhD", "DPhil", "Doctor of Philosophy"
          "field": string or null,           // e.g., "History", "Sociology", "Science and Technology Studies"; can be null for JD
          "institution": string or null,
          "year": string or null,
          "sources": [url, ...]              // URLs explicitly present in the answer that support this degree
        },
        ...
      ],
      "law_dean": {
        "law_school": string or null,        // e.g., "UCLA School of Law"
        "university": string or null,        // parent university name, if obvious from the answer
        "start_date": string or null,        // free text as shown in the answer, e.g., "2015" or "July 2015"
        "end_date": string or null,          // as shown in the answer
        "duration_years": string or null,    // if the answer states a duration, capture it as-is (e.g., "7", "at least 5")
        "sources": [url, ...]                // URLs cited for the deanship and tenure
      } or null,
      "new_position": {
        "title": string or null,             // "Chancellor" or "President" (or equivalent system head)
        "university": string or null,        // destination institution name
        "announcement_date": string or null, // as shown in the answer
        "effective_date": string or null,    // as shown in the answer
        "sources": [url, ...]                // URLs cited for this appointment
      } or null,
      "law_school_r1_sources": [url, ...],   // URLs cited about the law dean's parent university being R1 or equivalent
      "destination_inst_type_sources": [url, ...], // URLs cited about the destination university being flagship or R1
      "faculty_positions": [
        {
          "university": string or null,
          "school_or_dept": string or null,
          "role_title": string or null,
          "is_law_faculty": boolean or null, // true if explicitly a law faculty appointment; otherwise false or null
          "start_date": string or null,
          "end_date": string or null,
          "sources": [url, ...]              // URLs cited for this faculty role
        },
        ...
      ]
    }

    Rules:
    - Extract ONLY what is explicitly in the answer; do not invent.
    - For URLs, extract only valid, complete URLs that appear in the answer (plain or markdown). If none are provided for a fact, return an empty list for that 'sources' field.
    - If a value is not present, set it to null.
    - Prefer the most specific and relevant URLs for each field (e.g., official announcements, university bios/CV pages).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _collect_sources(*seqs: Optional[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for seq in seqs:
        if not seq:
            continue
        for u in seq:
            if not u:
                continue
            url = u.strip()
            if url and url not in seen:
                seen.add(url)
                out.append(url)
    return out


def _is_jd(deg: Degree) -> bool:
    dt = _lower(deg.degree_type)
    return any(k in dt for k in ["jd", "j.d", "juris doctor"])


def _is_phd_equiv(deg: Degree) -> bool:
    dt = _lower(deg.degree_type)
    return any(k in dt for k in ["phd", "ph.d", "doctor of philosophy", "dphil", "d.phil"])


def _any_law_faculty(fp: List[FacultyAppointment]) -> Optional[FacultyAppointment]:
    # Prefer explicit flags; fallback to textual heuristics
    for appt in fp:
        if appt.is_law_faculty is True:
            return appt
    for appt in fp:
        text = f"{appt.school_or_dept or ''} {appt.role_title or ''}".lower()
        if "law" in text and any(k in text for k in ["professor", "lecturer", "faculty", "associate", "assistant", "adjunct", "chair"]):
            return appt
    return None


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def verify_educational_credentials(evaluator: Evaluator, parent_node, ex: CandidateExtraction) -> None:
    name = ex.full_name or "the identified person"

    edu_node = evaluator.add_parallel(
        id="Educational_Credentials",
        desc="Verify the identified person holds the required educational degrees",
        parent=parent_node,
        critical=True
    )

    # JD
    jd_degrees = [d for d in ex.degrees if _is_jd(d)]
    jd_sources = _collect_sources(*[d.sources for d in jd_degrees])
    evaluator.add_custom_node(
        result=bool(jd_sources),
        id="JD_Sources_Provided",
        desc="URLs supporting the JD degree are provided in the answer",
        parent=edu_node,
        critical=True
    )

    jd_leaf = evaluator.add_leaf(
        id="JD_Degree",
        desc="Person holds a Juris Doctor (JD) degree from an accredited U.S. law school",
        parent=edu_node,
        critical=True
    )
    if jd_degrees:
        j0 = jd_degrees[0]
        claim_jd = f"{name} holds a Juris Doctor (JD) degree" + (f" from {j0.institution}." if j0.institution else ".")
    else:
        claim_jd = f"{name} holds a Juris Doctor (JD) degree."
    await evaluator.verify(
        claim=claim_jd,
        node=jd_leaf,
        sources=jd_sources,
        additional_instruction=(
            "Confirm that the page explicitly states the person earned a Juris Doctor (J.D.) or equivalent professional law degree. "
            "Do NOT accept research law doctorates such as SJD/JSD alone as JD. Prefer official bios or CVs."
        )
    )

    # PhD (non-law field)
    phd_degrees = [d for d in ex.degrees if _is_phd_equiv(d)]
    phd_sources = _collect_sources(*[d.sources for d in phd_degrees])
    evaluator.add_custom_node(
        result=bool(phd_sources),
        id="PhD_Sources_Provided",
        desc="URLs supporting the PhD (or DPhil) degree are provided in the answer",
        parent=edu_node,
        critical=True
    )

    phd_leaf = evaluator.add_leaf(
        id="PhD_Degree_in_Approved_Field",
        desc="Person holds a PhD degree in a field outside of pure law, such as history, sociology, science studies, or technology studies",
        parent=edu_node,
        critical=True
    )
    if phd_degrees:
        p0 = phd_degrees[0]
        fld = p0.field or "a non-law field"
        inst = f" from {p0.institution}" if p0.institution else ""
        claim_phd = f"{name} holds a PhD (or DPhil) in {fld}{inst}, and the field is outside of pure law."
    else:
        claim_phd = f"{name} holds a PhD (or DPhil) in a field outside of pure law."
    await evaluator.verify(
        claim=claim_phd,
        node=phd_leaf,
        sources=phd_sources,
        additional_instruction=(
            "Verify that the person earned a research doctorate (PhD/DPhil) and that the field is clearly not pure law. "
            "Accept fields like History, Sociology, Science and Technology Studies (HASTS/STS), Political Science, Economics, Philosophy (non-legal). "
            "Do NOT accept 'PhD in Law', 'Legal Studies' as pure-law fields. DPhil is equivalent to PhD."
        )
    )


async def verify_law_school_dean_experience(evaluator: Evaluator, parent_node, ex: CandidateExtraction) -> None:
    name = ex.full_name or "the identified person"
    dean_node = evaluator.add_parallel(
        id="Law_School_Dean_Experience",
        desc="Verify the person served as dean of a law school meeting specific criteria",
        parent=parent_node,
        critical=True
    )

    dean = ex.law_dean or DeanRole()
    dean_sources = _collect_sources(dean.sources)
    evaluator.add_custom_node(
        result=bool(dean_sources),
        id="Dean_Sources_Provided",
        desc="URLs supporting the law school deanship are provided in the answer",
        parent=dean_node,
        critical=True
    )

    # Dean position held
    dean_pos_leaf = evaluator.add_leaf(
        id="Dean_Position_Held",
        desc="Person served as dean of a law school at a U.S. university",
        parent=dean_node,
        critical=True
    )
    ls = dean.law_school or "a law school"
    univ = dean.university
    add_inst = "Confirm the page explicitly states service as dean of a law school."
    if univ:
        claim_dean_pos = f"{name} served as dean of {ls} at {univ}."
    else:
        claim_dean_pos = f"{name} served as dean of {ls}."
    await evaluator.verify(
        claim=claim_dean_pos,
        node=dean_pos_leaf,
        sources=dean_sources,
        additional_instruction=add_inst
    )

    # Dean tenure duration (>= 5 years)
    dean_tenure_leaf = evaluator.add_leaf(
        id="Dean_Tenure_Duration",
        desc="Person served as law school dean for at least 5 years",
        parent=dean_node,
        critical=True
    )
    sd = dean.start_date or ""
    ed = dean.end_date or ""
    claim_tenure = (
        f"{name} served as law school dean for at least five years"
        + (f", from {sd} to {ed}." if sd or ed else ".")
    )
    await evaluator.verify(
        claim=claim_tenure,
        node=dean_tenure_leaf,
        sources=dean_sources,
        additional_instruction=(
            "Use the dates on the page to determine the duration. If only years are given, treat the difference in years as the duration. "
            "Count interim/acting time if clearly indicated as dean. Pass if the tenure totals 5 or more years."
        )
    )

    # Law school institution type (R1 or equivalent)
    r1_sources = _collect_sources(ex.law_school_r1_sources, dean_sources)
    law_type_leaf = evaluator.add_leaf(
        id="Law_School_Institution_Type",
        desc="The law school was at an R1 (Research 1) university or equivalent major institution",
        parent=dean_node,
        critical=True
    )
    univ_name = dean.university or "the parent university"
    claim_r1 = f"{univ_name} is an R1 (Very high research activity) university or an equivalent major research institution."
    await evaluator.verify(
        claim=claim_r1,
        node=law_type_leaf,
        sources=r1_sources,
        additional_instruction=(
            "Accept explicit 'R1: Very high research activity' per the Carnegie Classification or an equivalent major-research designation. "
            "Also accept clear evidence that it is a flagship state university. Prefer official sites or authoritative references."
        )
    )


async def verify_university_leadership_transition(evaluator: Evaluator, parent_node, ex: CandidateExtraction) -> None:
    name = ex.full_name or "the identified person"
    trans_node = evaluator.add_parallel(
        id="University_Leadership_Transition",
        desc="Verify the career transition meets all specified requirements",
        parent=parent_node,
        critical=True
    )

    newpos = ex.new_position or Appointment()
    appt_sources = _collect_sources(newpos.sources)
    evaluator.add_custom_node(
        result=bool(appt_sources),
        id="Appointment_Sources_Provided",
        desc="URLs supporting the chancellor/president appointment are provided in the answer",
        parent=trans_node,
        critical=True
    )

    # New position type (chancellor or president, at a university)
    newpos_leaf = evaluator.add_leaf(
        id="New_Position_Type",
        desc="Person was appointed as chancellor or president of a university (not just a college or school)",
        parent=trans_node,
        critical=True
    )
    title = newpos.title or "a university leadership role"
    dst_univ = newpos.university or "a university"
    claim_newpos = f"{name} was appointed as {title} of {dst_univ}."
    await evaluator.verify(
        claim=claim_newpos,
        node=newpos_leaf,
        sources=appt_sources,
        additional_instruction=(
            "Confirm the role is university-level: 'President' or 'Chancellor' (including campus chancellor in multi-campus systems). "
            "Do NOT accept dean, provost, vice president, or school-level heads."
        )
    )

    # Different institution (destination ≠ dean's university) - logical check
    diff_leaf = evaluator.add_leaf(
        id="Different_Institution",
        desc="The new chancellor/president position is at a different institution than where the person served as law school dean",
        parent=trans_node,
        critical=True
    )
    dean_univ = (ex.law_dean.university if ex.law_dean else None) or ""
    dst_univ = newpos.university or ""
    claim_diff = (
        f"The destination institution '{dst_univ}' is different from the law dean's institution '{dean_univ}'. "
        "Treat differences in campus names or system vs. campus names carefully; consider true if they are clearly different institutions."
    )
    await evaluator.verify(
        claim=claim_diff,
        node=diff_leaf,
        sources=None,
        additional_instruction=(
            "Make a logical comparison of the two institution names (case-insensitive). "
            "If they clearly denote different universities/campuses, mark as correct."
        )
    )

    # Appointment timeline within 2022-01-01 to 2026-12-31 (inclusive)
    time_leaf = evaluator.add_leaf(
        id="Appointment_Timeline",
        desc="The appointment announcement or effective date occurred between January 1, 2022, and December 31, 2026 (inclusive)",
        parent=trans_node,
        critical=True
    )
    ann = newpos.announcement_date or ""
    eff = newpos.effective_date or ""
    claim_time = (
        f"The appointment (announcement and/or effective date) occurred within 2022-01-01 through 2026-12-31 inclusive. "
        f"Announcement date on record: '{ann}'. Effective date on record: '{eff}'."
    )
    await evaluator.verify(
        claim=claim_time,
        node=time_leaf,
        sources=appt_sources,
        additional_instruction=(
            "Pass if EITHER the announcement date OR the effective/start date on the page falls within 2022-01-01 and 2026-12-31 inclusive."
        )
    )

    # Destination institution type (flagship or R1)
    dst_type_sources = _collect_sources(ex.destination_inst_type_sources, appt_sources)
    dst_type_leaf = evaluator.add_leaf(
        id="Destination_Institution_Type",
        desc="The new institution is a flagship state university or R1 research university",
        parent=trans_node,
        critical=True
    )
    dst_univ = newpos.university or "the destination university"
    claim_dst_type = f"{dst_univ} is either a flagship state university or an R1 research university."
    await evaluator.verify(
        claim=claim_dst_type,
        node=dst_type_leaf,
        sources=dst_type_sources,
        additional_instruction=(
            "Accept explicit statements that the university is the state's 'flagship' campus or has 'R1: Very high research activity' classification. "
            "Prefer official or authoritative references."
        )
    )


async def verify_prior_faculty_experience(evaluator: Evaluator, parent_node, ex: CandidateExtraction) -> None:
    name = ex.full_name or "the identified person"
    fac_node = evaluator.add_parallel(
        id="Prior_Faculty_Experience",
        desc="Verify the person has the required faculty background",
        parent=parent_node,
        critical=True
    )

    # Gather faculty sources
    faculty_sources = _collect_sources(*[fp.sources for fp in ex.faculty_positions])
    evaluator.add_custom_node(
        result=bool(faculty_sources),
        id="Faculty_Sources_Provided",
        desc="URLs supporting faculty positions are provided in the answer",
        parent=fac_node,
        critical=True
    )

    # Multiple university appointments (>= 2 universities)
    multi_leaf = evaluator.add_leaf(
        id="Multiple_University_Appointments",
        desc="Person held faculty positions at at least two different universities before becoming dean",
        parent=fac_node,
        critical=True
    )
    # Build a readable list for the claim (best-effort)
    uniq_univs = []
    seen = set()
    for fp in ex.faculty_positions:
        if fp.university:
            u = fp.university.strip()
            if u and u.lower() not in seen:
                uniq_univs.append(u)
                seen.add(u.lower())
    example_list = ", ".join(uniq_univs[:3]) if uniq_univs else ""
    claim_multi = (
        f"{name} held faculty positions at at least two different universities."
        + (f" Examples include: {example_list}." if example_list else "")
    )
    await evaluator.verify(
        claim=claim_multi,
        node=multi_leaf,
        sources=faculty_sources,
        additional_instruction=(
            "A 'faculty position' includes professor/associate/assistant professor, lecturer, adjunct, visiting professor, or named chair roles. "
            "Administrative-only roles (e.g., pure staff) do not count. Pass if evidence shows two or more distinct universities where the person held a faculty role."
        )
    )

    # Law faculty position
    law_fac = _any_law_faculty(ex.faculty_positions)
    law_fac_leaf = evaluator.add_leaf(
        id="Law_Faculty_Position",
        desc="Person held a law faculty (law professor) position",
        parent=fac_node,
        critical=True
    )
    if law_fac:
        role = (law_fac.role_title or "a law faculty role").strip()
        sch = (law_fac.school_or_dept or "").strip()
        at_txt = f" at {sch}" if sch else ""
        claim_law = f"{name} held {role}{at_txt}, which is a law faculty appointment."
    else:
        claim_law = f"{name} held a law faculty appointment (such as Professor of Law, Associate Professor of Law, Adjunct Professor of Law, or Lecturer in Law)."
    await evaluator.verify(
        claim=claim_law,
        node=law_fac_leaf,
        sources=faculty_sources,
        additional_instruction=(
            "Accept explicit law-school faculty roles (Professor of Law, Associate/Assistant Professor of Law, Adjunct Professor of Law, Lecturer in Law, etc.). "
            "Being a law school dean can count if the page also indicates professorial status in the law school."
        )
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
    Evaluate an answer for the Academic Leader (JD + non-law PhD; dean to president/chancellor 2022–2026) task.
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
        default_model=model
    )

    # Extract structured candidate info
    extracted: CandidateExtraction = await evaluator.extract(
        prompt=prompt_extract_candidate(),
        template_class=CandidateExtraction,
        extraction_name="candidate_extraction"
    )

    # Build rubric tree under a critical top-level node
    al_node = evaluator.add_parallel(
        id="Academic_Leader_Identification",
        desc="Identify an academic leader who transitioned from law school dean to university chancellor/president between 2022-2026, with specific educational and career qualifications",
        parent=root,
        critical=True
    )

    # Subtrees (all critical per rubric)
    await verify_educational_credentials(evaluator, al_node, extracted)
    await verify_law_school_dean_experience(evaluator, al_node, extracted)
    await verify_university_leadership_transition(evaluator, al_node, extracted)
    await verify_prior_faculty_experience(evaluator, al_node, extracted)

    # Return structured result
    return evaluator.get_summary()