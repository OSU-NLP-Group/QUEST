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
TASK_ID = "law_deans_to_presidents_2020_2026"
TASK_DESCRIPTION = """
Identify at least three individuals who transitioned from law school dean positions to university president or chancellor positions between 2020 and 2026 (inclusive). For each individual, provide comprehensive verification of their career path including: (1) educational credentials (PhD, JD, and undergraduate degrees with institutions and years), with at least one degree from an Ivy League institution; (2) law faculty experience spanning at least 5 years before becoming dean; (3) intermediate administrative positions held before becoming dean; (4) law school dean experience of at least 5 years with institution and dates; (5) university president/chancellor appointment details including institution, appointment year, and start date; (6) primary area of legal scholarship. All information must be supported by reference URLs.
"""

IVY_LEAGUE_SET = {
    "harvard university",
    "yale university",
    "princeton university",
    "columbia university",
    "cornell university",
    "brown university",
    "dartmouth college",
    "university of pennsylvania",
    "upenn",
    "penn"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DegreeEntry(BaseModel):
    degree: Optional[str] = None   # e.g., JD, PhD, BA, AB, BS
    field: Optional[str] = None    # e.g., Law, Political Science
    institution: Optional[str] = None
    year: Optional[str] = None     # keep as string (can be "2005", "2004–2006", etc.)


class PositionEntry(BaseModel):
    title: Optional[str] = None
    institution: Optional[str] = None
    start: Optional[str] = None    # string date or year
    end: Optional[str] = None      # string date or "present"


class DeanRole(BaseModel):
    law_school: Optional[str] = None   # e.g., "XYZ School of Law"
    university: Optional[str] = None   # parent university if applicable
    start: Optional[str] = None
    end: Optional[str] = None


class LeaderRole(BaseModel):
    title: Optional[str] = None            # e.g., President, Chancellor
    institution: Optional[str] = None      # e.g., "University of X"
    announcement_date: Optional[str] = None  # year or full date string
    start_date: Optional[str] = None         # full date string where possible


class ScholarshipInfo(BaseModel):
    primary_area: Optional[str] = None     # e.g., constitutional law, IP law


class IndividualInfo(BaseModel):
    name: Optional[str] = None

    education: List[DegreeEntry] = Field(default_factory=list)
    education_urls: List[str] = Field(default_factory=list)

    faculty_positions: List[PositionEntry] = Field(default_factory=list)
    faculty_urls: List[str] = Field(default_factory=list)

    admin_positions: List[PositionEntry] = Field(default_factory=list)
    admin_urls: List[str] = Field(default_factory=list)

    dean: Optional[DeanRole] = None
    dean_urls: List[str] = Field(default_factory=list)

    leader: Optional[LeaderRole] = None
    leader_urls: List[str] = Field(default_factory=list)

    scholarship: Optional[ScholarshipInfo] = None
    scholarship_urls: List[str] = Field(default_factory=list)


class PeopleExtraction(BaseModel):
    individuals: List[IndividualInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_people() -> str:
    return """
    You must extract structured information about individuals described in the answer who transitioned from a law school dean position to a university president or chancellor position.
    Extract up to 5 individuals exactly as stated in the answer (do not invent). For each person, extract:

    - name: Full name of the individual.

    - education: An array of objects. Each object has:
        • degree: Degree name (e.g., "JD", "PhD", "BA", "AB", "BS"; use common abbreviations if available exactly as in the answer).
        • field: Optional field/major (e.g., "Law", "Political Science") if present.
        • institution: Granting institution as named in the answer.
        • year: The year (or year range if given) of completion; if missing, set to null.
      Also collect education_urls: all URLs in the answer that substantiate degrees, institutions, and years (profiles, CVs, news releases, etc.).

    - faculty_positions: An array of objects representing pre-dean law faculty roles (e.g., Assistant/Associate/Full Professor of Law). Each has:
        • title, institution, start, end.
      Collect faculty_urls: URLs that substantiate these faculty roles and dates.

    - admin_positions: An array of pre-dean administrative roles (e.g., Associate Dean, Vice Dean, Department Chair, Center Director) with:
        • title, institution, start, end.
      Collect admin_urls: URLs that substantiate these roles and dates.

    - dean: The deanship role object:
        • law_school: Name of the law school (e.g., "ABC School of Law").
        • university: Parent university if mentioned.
        • start: Start date or year of deanship.
        • end: End date or year of deanship (or "present" if applicable).
      Collect dean_urls: URLs that substantiate the deanship institution and dates.

    - leader: The university leader role object:
        • title: "President" or "Chancellor" (or closely equivalent).
        • institution: Name of the university.
        • announcement_date: Appointment announcement date (or year) if provided.
        • start_date: Start date if provided.
      Collect leader_urls: URLs that substantiate the appointment details and relevant dates.

    - scholarship: The legal scholarship information object:
        • primary_area: The individual's primary area of legal scholarship (e.g., "constitutional law", "environmental law").
      Collect scholarship_urls: URLs that substantiate the primary area and/or publications in that field (e.g., faculty profile, publications list, Google Scholar, SSRN).

    GENERAL RULES:
    1) Return only information explicitly present in the answer. If a field is missing, set it to null (or empty array for URL lists).
    2) For all *_urls fields, include only actual URLs explicitly present in the answer text.
    3) Preserve the textual form of dates/years as given (e.g., "2015", "2014–2019", "July 1, 2021").
    4) Do not infer or invent any data beyond what is stated in the answer.

    Return a JSON object of the form:
    {
      "individuals": [
        {
          "name": ...,
          "education": [ { "degree": ..., "field": ..., "institution": ..., "year": ... }, ... ],
          "education_urls": [ ... ],
          "faculty_positions": [ { "title": ..., "institution": ..., "start": ..., "end": ... }, ... ],
          "faculty_urls": [ ... ],
          "admin_positions": [ { "title": ..., "institution": ..., "start": ..., "end": ... }, ... ],
          "admin_urls": [ ... ],
          "dean": { "law_school": ..., "university": ..., "start": ..., "end": ... },
          "dean_urls": [ ... ],
          "leader": { "title": ..., "institution": ..., "announcement_date": ..., "start_date": ... },
          "leader_urls": [ ... ],
          "scholarship": { "primary_area": ... },
          "scholarship_urls": [ ... ]
        },
        ...
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _lc(s: Optional[str]) -> str:
    return _norm(s).lower()


def find_degree(entries: List[DegreeEntry], targets: List[str]) -> Optional[DegreeEntry]:
    targets_lc = {t.lower() for t in targets}
    for d in entries:
        deg = _lc(d.degree)
        if not deg:
            continue
        # Allow common synonyms/variants
        deg_clean = deg.replace(".", "").replace(" ", "")
        # Mapping buckets
        if any(t in deg for t in targets_lc) or any(t.replace(".", "").replace(" ", "") in deg_clean for t in targets_lc):
            return d
        # Handle some common equivalences:
        if "ab" in targets_lc and deg_clean in {"ba", "ab"}:
            return d
        if "ba" in targets_lc and deg_clean in {"ba", "ab"}:
            return d
        if "bs" in targets_lc and deg_clean in {"bs", "bsc", "sb"}:
            return d
        if "phd" in targets_lc and ("phd" in deg_clean or "dphil" in deg_clean):
            return d
        if "jd" in targets_lc and ("jd" in deg_clean or "j d" in deg_clean or "jurisdoctor" in deg_clean):
            return d
    return None


def has_ivy_degree(education: List[DegreeEntry]) -> bool:
    for d in education:
        inst = _lc(d.institution)
        if inst in IVY_LEAGUE_SET:
            return True
        # allow common shortened forms
        if "harvard" in inst or "yale" in inst or "princeton" in inst or "columbia" in inst or \
           "cornell" in inst or "dartmouth" in inst or "brown" in inst or \
           inst in {"university of pennsylvania", "upenn", "penn"}:
            # still validate with broader contains
            return True
    return False


def degree_institutions_summary(education: List[DegreeEntry]) -> str:
    parts = []
    for d in education:
        deg = _norm(d.degree)
        inst = _norm(d.institution)
        year = _norm(d.year)
        if deg or inst or year:
            parts.append(f"{deg or 'Unknown degree'} at {inst or 'Unknown institution'} ({year or 'year unknown'})")
    return "; ".join(parts) if parts else "None provided"


def list_unique_institutions(education: List[DegreeEntry]) -> List[str]:
    insts = []
    seen = set()
    for d in education:
        inst = _norm(d.institution)
        if inst and inst.lower() not in seen:
            seen.add(inst.lower())
            insts.append(inst)
    return insts


def positions_summary(positions: List[PositionEntry]) -> str:
    items = []
    for p in positions:
        items.append(f"{_norm(p.title) or 'Unknown title'} at {_norm(p.institution) or 'Unknown institution'} ({_norm(p.start) or '?'} – {_norm(p.end) or '?'})")
    return "; ".join(items) if items else "None provided"


def dean_summary(dean: Optional[DeanRole]) -> str:
    if not dean:
        return "None provided"
    return f"Dean at {_norm(dean.law_school) or 'Unknown law school'} ({_norm(dean.university) or 'unknown university'}) from {_norm(dean.start) or '?'} to {_norm(dean.end) or '?'}"


def leader_summary(leader: Optional[LeaderRole]) -> str:
    if not leader:
        return "None provided"
    return f"{_norm(leader.title) or 'Unknown role'} at {_norm(leader.institution) or 'Unknown institution'}; announcement: {_norm(leader.announcement_date) or 'unknown'}; start: {_norm(leader.start_date) or 'unknown'}"


def urls_or_none(urls: Optional[List[str]]) -> List[str]:
    return urls or []


# --------------------------------------------------------------------------- #
# Verification for one individual                                             #
# --------------------------------------------------------------------------- #
async def verify_individual(
    evaluator: Evaluator,
    root_parent,
    person: IndividualInfo,
    idx: int,
) -> str:
    """
    Build tree and run verifications for one individual.
    Returns the node ID for the individual's top-level node.
    """
    ind_num = idx + 1
    ind_node = evaluator.add_parallel(
        id=f"Individual_{ind_num}",
        desc=f"Individual #{ind_num} (qualifying person).",
        parent=root_parent,
        critical=False
    )

    # 1) Name existence (critical)
    name_ok = bool(_norm(person.name))
    evaluator.add_custom_node(
        result=name_ok,
        id=f"Ind{ind_num}_Name",
        desc="Individual’s identifying name is provided.",
        parent=ind_node,
        critical=True
    )

    # 2) Education (critical parallel)
    edu_node = evaluator.add_parallel(
        id=f"Ind{ind_num}_Education",
        desc="Education meets degree/provenance constraints with institutions and years.",
        parent=ind_node,
        critical=True
    )

    # Helper: locate PhD, JD, and an undergraduate degree
    phd = find_degree(person.education, ["PhD", "DPhil"])
    jd = find_degree(person.education, ["JD", "J.D.", "Juris Doctor"])
    ug = find_degree(person.education, ["BA", "AB", "BS", "BSc", "Bachelor"])

    # 2.1 Degrees listed with years (verify against education URLs)
    leaf_deg_years = evaluator.add_leaf(
        id=f"Ind{ind_num}_Degrees_Listed_With_Years",
        desc="PhD, JD, and undergraduate degree are provided with granting institutions and years.",
        parent=edu_node,
        critical=True
    )
    claim_deg_years = (
        f"The person has the following three degrees with institutions and years explicitly supported by the sources: "
        f"PhD: institution='{_norm(phd.institution) if phd else ''}', year='{_norm(phd.year) if phd else ''}'; "
        f"JD: institution='{_norm(jd.institution) if jd else ''}', year='{_norm(jd.year) if jd else ''}'; "
        f"Undergraduate: degree='{_norm(ug.degree) if ug else ''}', institution='{_norm(ug.institution) if ug else ''}', year='{_norm(ug.year) if ug else ''}'. "
        f"All three (PhD, JD, and an undergraduate degree) must be present with institutions and years to be considered supported."
    )
    await evaluator.verify(
        claim=claim_deg_years,
        node=leaf_deg_years,
        sources=urls_or_none(person.education_urls),
        additional_instruction="Treat AB as equivalent to BA; DPhil as equivalent to PhD. If any of the three degrees or their years are missing or not clearly supported by the provided sources, mark as not supported."
    )

    # 2.2 PhD/JD accredited institutions
    leaf_accred = evaluator.add_leaf(
        id=f"Ind{ind_num}_PhD_JD_Accredited",
        desc="The PhD-granting institution and the JD-granting institution are accredited with sufficient evidence.",
        parent=edu_node,
        critical=True
    )
    claim_accred = (
        f"The PhD-granting institution '{_norm(phd.institution) if phd else ''}' and the JD-granting institution "
        f"'{_norm(jd.institution) if jd else ''}' are accredited (recognized universities / ABA-approved law schools), as supported by the sources."
    )
    await evaluator.verify(
        claim=claim_accred,
        node=leaf_accred,
        sources=urls_or_none(person.education_urls),
        additional_instruction="Verify that the PhD university is a recognized accredited higher-education institution and the JD comes from a recognized/ABA-approved law school. If unclear, mark as not supported."
    )

    # 2.3 Ivy League check
    leaf_ivy = evaluator.add_leaf(
        id=f"Ind{ind_num}_Ivy_League_Degree",
        desc="At least one degree (undergrad, JD, or PhD) is from an Ivy League institution.",
        parent=edu_node,
        critical=True
    )
    degrees_list_txt = degree_institutions_summary(person.education)
    claim_ivy = (
        f"At least one of the degrees is from an Ivy League institution (Harvard, Yale, Princeton, Columbia, Cornell, Brown, Dartmouth, University of Pennsylvania). "
        f"Degrees listed: {degrees_list_txt}."
    )
    await evaluator.verify(
        claim=claim_ivy,
        node=leaf_ivy,
        sources=urls_or_none(person.education_urls),
        additional_instruction="Use the sources to confirm that at least one degree institution is one of the Ivy League schools. Consider 'UPenn' or 'Penn' as University of Pennsylvania."
    )

    # 2.4 Degrees from at least two different institutions
    leaf_two_inst = evaluator.add_leaf(
        id=f"Ind{ind_num}_Two_Institutions_Minimum",
        desc="Degrees come from at least two different institutions.",
        parent=edu_node,
        critical=True
    )
    insts = list_unique_institutions(person.education)
    claim_two_inst = (
        f"The degrees were earned from at least two different institutions. The unique institutions listed are: {', '.join(insts) if insts else 'none'}."
    )
    await evaluator.verify(
        claim=claim_two_inst,
        node=leaf_two_inst,
        sources=urls_or_none(person.education_urls),
        additional_instruction="Confirm from sources that there are minimally two distinct institutions across the listed degrees."
    )

    # 3) Pre-dean faculty experience ≥ 5 years (critical)
    leaf_fac = evaluator.add_leaf(
        id=f"Ind{ind_num}_PreDean_Faculty_Experience",
        desc="Before becoming dean, served as law faculty for ≥5 years (institutions and dates sufficient to verify).",
        parent=ind_node,
        critical=True
    )
    dean_start_txt = _norm(person.dean.start) if person.dean else ""
    claim_fac = (
        f"Before becoming dean (deanship start: '{dean_start_txt}'), the individual served as law faculty for at least 5 years. "
        f"Extracted faculty roles: {positions_summary(person.faculty_positions)}."
    )
    await evaluator.verify(
        claim=claim_fac,
        node=leaf_fac,
        sources=urls_or_none(person.faculty_urls),
        additional_instruction="Confirm that cumulative service as law faculty (e.g., assistant/associate/full professor of law) before the deanship is 5 years or more, using the provided roles and dates."
    )

    # 4) Intermediate administrative role(s) before dean (critical)
    leaf_admin = evaluator.add_leaf(
        id=f"Ind{ind_num}_Intermediate_Admin_PreDean",
        desc="At least one intermediate administrative position pre-dean is provided (title, institution, dates).",
        parent=ind_node,
        critical=True
    )
    claim_admin = (
        f"Before becoming dean, the individual held at least one intermediate administrative position with title and dates (e.g., associate dean/vice dean/chair/director). "
        f"Extracted admin roles: {positions_summary(person.admin_positions)}."
    )
    await evaluator.verify(
        claim=claim_admin,
        node=leaf_admin,
        sources=urls_or_none(person.admin_urls),
        additional_instruction="The evidence should explicitly show a pre-dean administrative role with a title and timeframe."
    )

    # 5) Deanship ≥ 5 years at accredited university (critical)
    leaf_dean = evaluator.add_leaf(
        id=f"Ind{ind_num}_Dean_Experience",
        desc="Served as law school dean for ≥5 years; provide institution and dates sufficient to verify duration.",
        parent=ind_node,
        critical=True
    )
    claim_dean = (
        f"The individual served as dean of a law school for at least 5 years. {dean_summary(person.dean)}"
    )
    await evaluator.verify(
        claim=claim_dean,
        node=leaf_dean,
        sources=urls_or_none(person.dean_urls),
        additional_instruction="Confirm the deanship start and end (or current) dates indicate a tenure of 5 or more years, and that the institution is an accredited university law school."
    )

    # 6) President/Chancellor appointment details & timing (critical parallel)
    pres_node = evaluator.add_parallel(
        id=f"Ind{ind_num}_President_or_Chancellor",
        desc="University president/chancellor appointment details and timing constraints are satisfied.",
        parent=ind_node,
        critical=True
    )

    # 6.1 Details present
    leaf_pres_details = evaluator.add_leaf(
        id=f"Ind{ind_num}_Univ_Leader_Details",
        desc="President/chancellor role identified with institution, appointment/announcement year, and start date.",
        parent=pres_node,
        critical=True
    )
    claim_pres_details = (
        f"The person was appointed as {_norm(person.leader.title) if person.leader else ''} at "
        f"'{_norm(person.leader.institution) if person.leader else ''}', with announcement date/year "
        f"'{_norm(person.leader.announcement_date) if person.leader else ''}' and start date "
        f"'{_norm(person.leader.start_date) if person.leader else ''}', supported by sources."
    )
    await evaluator.verify(
        claim=claim_pres_details,
        node=leaf_pres_details,
        sources=urls_or_none(person.leader_urls),
        additional_instruction="The claim should be supported by official announcements or reputable news/university sources showing the leader role, institution, and dates."
    )

    # 6.2 Announcement or start in 2020–2026 inclusive
    leaf_pres_window = evaluator.add_leaf(
        id=f"Ind{ind_num}_AnnouncementOrStart_2020_2026",
        desc="Appointment announcement or start date is between Jan 1, 2020 and Dec 31, 2026 (inclusive).",
        parent=pres_node,
        critical=True
    )
    claim_pres_window = (
        f"The appointment announcement date '{_norm(person.leader.announcement_date) if person.leader else ''}' "
        f"or the start date '{_norm(person.leader.start_date) if person.leader else ''}' falls between 2020-01-01 and 2026-12-31 inclusive."
    )
    await evaluator.verify(
        claim=claim_pres_window,
        node=leaf_pres_window,
        sources=urls_or_none(person.leader_urls),
        additional_instruction="Use the provided dates to verify the window. Either the announcement or the start date within the 2020–2026 window is sufficient."
    )

    # 6.3 Transition timing relative to deanship (≤ 5-year gap)
    leaf_transition = evaluator.add_leaf(
        id=f"Ind{ind_num}_DeanToUnivLeader_Transition_Timing",
        desc="Leader appointment/start occurs after deanship and the gap is within 5 years or less.",
        parent=pres_node,
        critical=True
    )
    dean_end_txt = _norm(person.dean.end) if person.dean else ""
    claim_transition = (
        f"The university leader appointment or start occurs after the deanship ended and the gap is 5 years or less. "
        f"Deanship end: '{dean_end_txt}'. Leader announcement: '{_norm(person.leader.announcement_date) if person.leader else ''}'. "
        f"Leader start: '{_norm(person.leader.start_date) if person.leader else ''}'."
    )
    await evaluator.verify(
        claim=claim_transition,
        node=leaf_transition,
        sources=(urls_or_none(person.dean_urls) + urls_or_none(person.leader_urls)),
        additional_instruction="Cross-check the deanship end against the appointment/start dates. If dates are missing or indicate >5 years, mark as not supported."
    )

    # 7) Scholarship (critical parallel)
    schol_node = evaluator.add_parallel(
        id=f"Ind{ind_num}_Scholarship",
        desc="Legal scholarship requirement is satisfied.",
        parent=ind_node,
        critical=True
    )

    leaf_primary_sch = evaluator.add_leaf(
        id=f"Ind{ind_num}_Primary_Scholarship_Area",
        desc="Primary area of legal scholarship is stated.",
        parent=schol_node,
        critical=True
    )
    claim_primary_sch = (
        f"The individual's primary area of legal scholarship is '{_norm(person.scholarship.primary_area) if person.scholarship else ''}', supported by the sources."
    )
    await evaluator.verify(
        claim=claim_primary_sch,
        node=leaf_primary_sch,
        sources=urls_or_none(person.scholarship_urls),
        additional_instruction="Verify that the profile or publications clearly indicate the stated primary area."
    )

    leaf_schol_creds = evaluator.add_leaf(
        id=f"Ind{ind_num}_Scholarly_Credentials_Evidence",
        desc="Evidence of scholarly credentials in the stated field is provided.",
        parent=schol_node,
        critical=True
    )
    claim_schol_creds = (
        "The sources provide evidence of research/publications or recognized scholarly work in the stated legal field."
    )
    await evaluator.verify(
        claim=claim_schol_creds,
        node=leaf_schol_creds,
        sources=urls_or_none(person.scholarship_urls),
        additional_instruction="Look for publications lists, articles, Google Scholar/SSRN pages, or similar that align with the stated field."
    )

    # 8) References presence checks (critical parallel; existence only)
    refs_node = evaluator.add_parallel(
        id=f"Ind{ind_num}_References",
        desc="Public reference URLs collectively support all required facts for this individual.",
        parent=ind_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(urls_or_none(person.education_urls)) > 0,
        id=f"Ind{ind_num}_Ref_Education",
        desc="Education references (URL[s]) are provided.",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(len(urls_or_none(person.faculty_urls)) > 0 and len(urls_or_none(person.admin_urls)) > 0),
        id=f"Ind{ind_num}_Ref_PreDean_Faculty_And_Admin",
        desc="References for pre-dean faculty (with dates) and admin role(s) (with dates) are provided.",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(urls_or_none(person.dean_urls)) > 0,
        id=f"Ind{ind_num}_Ref_Dean",
        desc="References for the law school deanship facts are provided.",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(urls_or_none(person.leader_urls)) > 0,
        id=f"Ind{ind_num}_Ref_PresidentChancellor",
        desc="References for the president/chancellor appointment details are provided.",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(urls_or_none(person.scholarship_urls)) > 0,
        id=f"Ind{ind_num}_Ref_Scholarship",
        desc="References for the stated scholarship area/evidence are provided.",
        parent=refs_node,
        critical=True
    )

    return ind_node.id


# --------------------------------------------------------------------------- #
# Global qualification check                                                  #
# --------------------------------------------------------------------------- #
def count_fully_qualified(evaluator: Evaluator, individual_node_ids: List[str]) -> int:
    """
    Count how many individual nodes have all their critical descendants passed.
    We rely on aggregated scoring semantics: a parent with any failing critical child will aggregate to 0.0.
    """
    if not evaluator.root:
        return 0

    # Force computation and write-back of scores/statuses
    evaluator.root.compute_score(mutate=True)

    qualified = 0
    for nid in individual_node_ids:
        node = evaluator.find_node(nid)
        if not node:
            continue
        # If any critical child failed, aggregated score should be 0.0; otherwise 1.0.
        # Because all essential items in our Individual node are marked critical, a score of 1.0 implies all mandatory checks passed.
        try:
            score = node.aggregated_score
        except Exception:
            score = node.score
        if score == 1.0:
            qualified += 1
    return qualified


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
    Evaluate an answer for the 'law_deans_to_presidents_2020_2026' task.
    """
    # Initialize evaluator with a parallel root
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

    # Record Ivy League list as custom info for transparency
    evaluator.add_custom_info(
        {
            "ivy_league": sorted(list(IVY_LEAGUE_SET))
        },
        info_type="constants",
        info_name="ivy_reference"
    )

    # Extract structured people info
    extracted: PeopleExtraction = await evaluator.extract(
        prompt=prompt_extract_people(),
        template_class=PeopleExtraction,
        extraction_name="people_extraction"
    )

    # Keep only the first 3 individuals as required; pad with empty if fewer
    persons: List[IndividualInfo] = list(extracted.individuals[:3])
    while len(persons) < 3:
        persons.append(IndividualInfo())

    # Build verification trees for each individual
    individual_node_ids: List[str] = []
    for i in range(3):
        node_id = await verify_individual(evaluator, root, persons[i], i)
        individual_node_ids.append(node_id)

    # After building/verifying individuals, add the global minimum qualified count check (critical)
    num_qualified = count_fully_qualified(evaluator, individual_node_ids)
    evaluator.add_custom_node(
        result=(num_qualified >= 3),
        id="Global_Minimum_Qualified_Individuals",
        desc="At least three individuals in the response satisfy ALL mandatory (critical) criteria for an individual.",
        parent=root,
        critical=True
    )

    # Return final structured summary
    return evaluator.get_summary()