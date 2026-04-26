import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "power5_president_long_tenure"
TASK_DESCRIPTION = """Identify a person who currently serves as president or chancellor of a major research university in the United States and who meets ALL of the following criteria:

1. Previously served as dean of a law school, business school, or engineering college at a U.S. university
2. Holds a doctoral-level degree (Ph.D., J.D., or equivalent terminal degree)
3. Has published scholarly research or academic work in their field of expertise
4. Currently leads a university that is a member of a Power Five athletic conference (ACC, Big Ten, SEC, Big 12, or Pac-12)
5. Has served in their current presidential or chancellor role for at least 10 consecutive years as of 2026

Provide the following information about this individual:
- Full name and current official title
- Current university name
- Previous dean position(s) including the specific school/college and university name(s)
- Doctoral degree information (degree type, field of study, granting institution)
- The athletic conference of their current university
- The year they began their current presidential/chancellor role
- Reference URLs from official university sources or reputable news organizations supporting the key facts
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CurrentPosition(BaseModel):
    title: Optional[str] = None
    university: Optional[str] = None
    conference: Optional[str] = None
    start_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    conference_sources: List[str] = Field(default_factory=list)
    start_year_sources: List[str] = Field(default_factory=list)


class DeanPosition(BaseModel):
    school_type: Optional[str] = None  # e.g., "law", "business", "engineering"
    school_name: Optional[str] = None  # e.g., "School of Law", "College of Engineering"
    university: Optional[str] = None
    title: Optional[str] = None  # e.g., "Dean", "Dean of Engineering"
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DegreeInfo(BaseModel):
    degree_type: Optional[str] = None  # e.g., "Ph.D.", "J.D.", "D.Phil."
    field: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ScholarshipInfo(BaseModel):
    description: Optional[str] = None  # brief summary or examples
    field: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PersonInfo(BaseModel):
    full_name: Optional[str] = None
    current_position: Optional[CurrentPosition] = None
    dean_positions: List[DeanPosition] = Field(default_factory=list)
    degree: Optional[DegreeInfo] = None
    scholarship: Optional[ScholarshipInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_person_info() -> str:
    return """
Extract the information for a single individual described in the answer (assume the answer focuses on one person; if multiple are listed, extract the first). Return a JSON object with the following fields:

- full_name: The individual's full name, exactly as written in the answer.
- current_position: An object with:
  - title: Current official title (e.g., President, Chancellor).
  - university: Current university name.
  - conference: The athletic conference for the current university, if explicitly stated.
  - start_year: The year they began their current presidential/chancellor role, as presented in the answer (string; may include month text).
  - sources: An array of URLs in the answer that confirm the current title and university (aim for official .edu pages or official bios; include all provided).
  - conference_sources: An array of URLs supporting the conference membership of the current university (conference/athletics/university pages or reputable news).
  - start_year_sources: An array of URLs that support the stated start year for the current role (official pages or reputable news).
- dean_positions: An array of objects; include all deanships mentioned in the answer. Each entry with:
  - school_type: One of "law", "business", "engineering" if stated or implied; otherwise extract the exact text (e.g., "law", "business school", "engineering college").
  - school_name: The specific school/college name (e.g., "School of Law", "College of Engineering").
  - university: The university where they served as dean.
  - title: The title held (e.g., "Dean", "Dean of Engineering").
  - start_year: The year the deanship started, if given (string).
  - end_year: The year the deanship ended, if given (string).
  - sources: URLs that support the deanship.
- degree: An object with:
  - degree_type: The doctoral-level or terminal degree (e.g., "Ph.D.", "J.D.", "D.Phil.", "Ed.D.", "M.D.", "Sc.D.", "DBA", "Eng.D.").
  - field: The field of study/specialization (if given).
  - institution: The granting institution.
  - year: The year awarded (if given; string).
  - sources: URLs that support the degree.
- scholarship: An object with:
  - description: Text in the answer that documents or summarizes academic publications, research, or scholarly contributions (e.g., named articles, books, research areas).
  - field: The field/area of scholarship if specified.
  - sources: URLs to publication lists, profiles, CVs, Google Scholar, or university pages that reference scholarly work.

Rules:
- Only extract information explicitly presented in the answer.
- For any missing field, return null (or empty list for arrays).
- For URLs, extract only valid URLs explicitly present in the answer. Include full protocol. Do not fabricate URLs.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
POWER5_CANONICAL = {
    "ACC": "ACC",
    "ATLANTIC COAST CONFERENCE": "ACC",
    "BIG TEN": "Big Ten",
    "SEC": "SEC",
    "SOUTHEASTERN CONFERENCE": "SEC",
    "BIG 12": "Big 12",
    "PAC-12": "Pac-12",
    "PAC 12": "Pac-12",
    "PAC12": "Pac-12",
}

REPUTABLE_NEWS_DOMAINS = {
    "nytimes.com", "washingtonpost.com", "wsj.com", "bloomberg.com", "reuters.com", "apnews.com",
    "bbc.com", "time.com", "usatoday.com", "theguardian.com", "forbes.com", "npr.org",
    "chronicle.com", "insidehighered.com", "abcnews.go.com", "cbsnews.com", "nbcnews.com", "cnn.com"
}

VALID_TERMINAL_DEGREES = {
    "PHD", "PH.D.", "PH D", "D.PHIL", "DPHIL",
    "JD", "J.D.", "J D", "SJD", "S.J.D.", "JSD", "J.S.D.",
    "MD", "M.D.", "ED.D.", "EDD", "SC.D.", "SCD",
    "DBA", "D.B.A.",
    "ENG.D.", "DENG", "D.ENG", "ENGD"
}


def first_int_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return int(m.group(0)) if m else None


def normalize_conference(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().upper()
    return POWER5_CANONICAL.get(s, POWER5_CANONICAL.get(s.replace("-", " "), None) or POWER5_CANONICAL.get(s.replace(" ", ""), None))


def is_power5_name(name: Optional[str]) -> bool:
    return normalize_conference(name) in {"ACC", "Big Ten", "SEC", "Big 12", "Pac-12"}


def domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        # Strip leading "www."
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def is_official_university_url(url: str) -> bool:
    d = domain_of(url)
    return d.endswith(".edu") or d.endswith(".edu/") or ".edu" in d


def is_credible_source(url: str) -> bool:
    d = domain_of(url)
    return is_official_university_url(url) or d.endswith(".gov") or any(d == dom or d.endswith("." + dom) for dom in REPUTABLE_NEWS_DOMAINS)


def is_valid_terminal_degree(degree_type: Optional[str]) -> bool:
    if not degree_type:
        return False
    s = degree_type.strip().upper().replace("–", "-")
    s = s.replace(" OF ", " ").replace(" IN ", " ")
    for deg in VALID_TERMINAL_DEGREES:
        if deg in s:
            return True
    return False


def allowed_dean_school_type(text: Optional[str]) -> bool:
    if not text:
        return False
    s = text.lower()
    return any(k in s for k in ["law", "business", "engineering"])


def pick_dean_position(positions: List[DeanPosition]) -> Optional[DeanPosition]:
    # Prefer positions with allowed school type and has at least one source
    for p in positions:
        if allowed_dean_school_type(p.school_type or "") and p.sources:
            return p
    for p in positions:
        if allowed_dean_school_type(p.school_type or ""):
            return p
    return positions[0] if positions else None


def unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        u = u.strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def aggregate_all_sources(person: PersonInfo) -> List[str]:
    urls: List[str] = []
    if person.current_position:
        urls += person.current_position.sources
        urls += person.current_position.conference_sources
        urls += person.current_position.start_year_sources
    for dp in person.dean_positions:
        urls += dp.sources
    if person.degree:
        urls += person.degree.sources
    if person.scholarship:
        urls += person.scholarship.sources
    return unique_urls(urls)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_person_identification(evaluator: Evaluator, parent, person: PersonInfo):
    node = evaluator.add_parallel(
        id="PersonIdentification",
        desc="Correct identification of an individual who currently serves as a university president or chancellor",
        parent=parent,
        critical=True
    )

    # Name provided (critical existence)
    evaluator.add_custom_node(
        result=bool(person.full_name and person.full_name.strip()),
        id="NameProvided",
        desc="Full name of the individual is provided",
        parent=node,
        critical=True
    )

    # Current university named (critical existence)
    current_univ = person.current_position.university if person.current_position else None
    evaluator.add_custom_node(
        result=bool(current_univ and current_univ.strip()),
        id="CurrentUniversityNamed",
        desc="Name of the current university is provided",
        parent=node,
        critical=True
    )

    # Reference URL for current position exists (must be official university source)
    current_sources = (person.current_position.sources if person.current_position else []) or []
    has_official_current = any(is_official_university_url(u) for u in current_sources)
    ref_current = evaluator.add_custom_node(
        result=has_official_current,
        id="ReferenceURL_CurrentPosition",
        desc="Official university source URL confirming current position is provided",
        parent=node,
        critical=True
    )

    # Current title accurate (verify with URL)
    title = person.current_position.title if person.current_position else None
    title_leaf = evaluator.add_leaf(
        id="CurrentTitleAccurate",
        desc="Current official title (president or chancellor) is correctly stated",
        parent=node,
        critical=True
    )
    claim_title = f"According to the provided official sources, {person.full_name} currently holds the official title '{title}' at {current_univ}."
    await evaluator.verify(
        claim=claim_title,
        node=title_leaf,
        sources=current_sources,
        additional_instruction="Verify the exact current title on the official university page(s). Allow reasonable formatting variants (e.g., 'President and Chancellor' variants).",
        extra_prerequisites=[ref_current]
    )

    # Current position verified (verify with URL)
    pos_leaf = evaluator.add_leaf(
        id="CurrentPositionVerified",
        desc="The individual currently holds the stated position at the stated university",
        parent=node,
        critical=True
    )
    claim_pos = f"{person.full_name} currently serves as {title} at {current_univ}."
    await evaluator.verify(
        claim=claim_pos,
        node=pos_leaf,
        sources=current_sources,
        additional_instruction="Confirm that the person currently holds the stated role at the stated university, based on the official page(s).",
        extra_prerequisites=[ref_current]
    )


async def verify_previous_deanship(evaluator: Evaluator, parent, person: PersonInfo):
    node = evaluator.add_sequential(
        id="PreviousDeanship",
        desc="Verification that the individual previously served as dean of a law school, business school, or engineering college",
        parent=parent,
        critical=True
    )

    # Select a candidate deanship
    dp = pick_dean_position(person.dean_positions)

    # DeanPositionIdentified parallel
    dean_block = evaluator.add_parallel(
        id="DeanPositionIdentified",
        desc="At least one previous dean position is identified with complete details",
        parent=node,
        critical=True
    )

    # School type correct
    evaluator.add_custom_node(
        result=allowed_dean_school_type(dp.school_type if dp else None),
        id="SchoolTypeCorrect",
        desc="The school/college was a law school, business school, or engineering college",
        parent=dean_block,
        critical=True
    )

    # Dean university named
    evaluator.add_custom_node(
        result=bool(dp and dp.university and dp.university.strip()),
        id="DeanUniversityNamed",
        desc="The university where they served as dean is named",
        parent=dean_block,
        critical=True
    )

    # Reference URL for deanship exists
    dp_sources = dp.sources if dp else []
    ref_dean = evaluator.add_custom_node(
        result=bool(dp_sources),
        id="ReferenceURL_DeanPosition",
        desc="Source URL confirming the dean position is provided",
        parent=dean_block,
        critical=True
    )

    # Dean title confirmed (URL verification)
    dean_title_leaf = evaluator.add_leaf(
        id="DeanTitleConfirmed",
        desc="The position held was specifically a dean position",
        parent=dean_block,
        critical=True
    )
    school_desc = dp.school_name or (dp.school_type or "the school/college")
    claim_dean = f"{person.full_name} served as a dean of {school_desc} at {dp.university if dp else ''}."
    await evaluator.verify(
        claim=claim_dean,
        node=dean_title_leaf,
        sources=dp_sources,
        additional_instruction="Verify that the person held a dean title for the specified school or college at the stated university.",
        extra_prerequisites=[ref_dean]
    )

    # Deanship precedes current role (sequential requirement, critical)
    precedes_leaf = evaluator.add_leaf(
        id="DeanshipPrecedesCurrentRole",
        desc="The dean position was held before the current presidential/chancellor role",
        parent=node,
        critical=True
    )
    start_year = person.current_position.start_year if person.current_position else None
    claim_precedes = f"Before becoming {person.current_position.title if person.current_position else 'the current role'} at {person.current_position.university if person.current_position else 'the current university'} in {start_year}, {person.full_name} served as dean at {dp.university if dp else ''}."
    # Use both deanship and current role sources
    combined_sources = dp_sources + (person.current_position.start_year_sources if person.current_position else []) + (person.current_position.sources if person.current_position else [])
    await evaluator.verify(
        claim=claim_precedes,
        node=precedes_leaf,
        sources=combined_sources,
        additional_instruction="Confirm the chronology that the deanship occurred before the current role began (use bios, announcements, CVs, or reputable news).",
        extra_prerequisites=[ref_dean]
    )


async def verify_doctoral_degree(evaluator: Evaluator, parent, person: PersonInfo):
    node = evaluator.add_parallel(
        id="DoctoralDegree",
        desc="Verification of doctoral-level degree credentials",
        parent=parent,
        critical=False  # Adjusted to allow a non-critical child within
    )

    deg = person.degree

    # Degree type valid (terminal/doctoral)
    evaluator.add_custom_node(
        result=is_valid_terminal_degree(deg.degree_type if deg else None),
        id="DegreeTypeValid",
        desc="The degree type is a recognized doctoral-level or equivalent terminal degree (Ph.D., J.D., or equivalent)",
        parent=node,
        critical=True
    )

    # Field of study provided (non-critical)
    evaluator.add_custom_node(
        result=bool(deg and deg.field and deg.field.strip()),
        id="FieldOfStudyProvided",
        desc="The field of study or specialization for the degree is provided",
        parent=node,
        critical=False
    )

    # Reference URL for degree exists
    degree_sources = deg.sources if deg else []
    ref_degree = evaluator.add_custom_node(
        result=bool(degree_sources),
        id="ReferenceURL_Degree",
        desc="Source URL confirming degree credentials is provided",
        parent=node,
        critical=True
    )

    # Granting institution correct (URL verification)
    inst_leaf = evaluator.add_leaf(
        id="GrantingInstitutionCorrect",
        desc="The university that granted the degree is correctly identified",
        parent=node,
        critical=True
    )
    claim_inst = f"{person.full_name} earned a {deg.degree_type if deg else ''} from {deg.institution if deg else ''}."
    await evaluator.verify(
        claim=claim_inst,
        node=inst_leaf,
        sources=degree_sources,
        additional_instruction="Verify that the stated institution awarded the specified degree to the person.",
        extra_prerequisites=[ref_degree]
    )


async def verify_scholarly_work(evaluator: Evaluator, parent, person: PersonInfo):
    node = evaluator.add_parallel(
        id="ScholarlyWork",
        desc="Evidence that the individual has published scholarly research or academic work",
        parent=parent,
        critical=False  # Adjusted to allow a non-critical child
    )

    sch = person.scholarship

    # Reference URL for scholarship exists
    sch_sources = sch.sources if sch else []
    ref_sch = evaluator.add_custom_node(
        result=bool(sch_sources),
        id="ReferenceURL_Scholarship",
        desc="Source URL referencing scholarly work or academic credentials is provided",
        parent=node,
        critical=True
    )

    # Scholarship documented (URL verification)
    sch_leaf = evaluator.add_leaf(
        id="ScholarshipDocumented",
        desc="Documentation or description of academic publications, research, or scholarly contributions is provided",
        parent=node,
        critical=True
    )
    claim_sch = f"{person.full_name} has published scholarly research or academic work{(' in ' + sch.field) if (sch and sch.field) else ''}."
    await evaluator.verify(
        claim=claim_sch,
        node=sch_leaf,
        sources=sch_sources,
        additional_instruction="Verify via publication list, CV, Google Scholar, or official university profile that the person has scholarly publications or academic work.",
        extra_prerequisites=[ref_sch]
    )

    # Field of scholarship identified (non-critical)
    evaluator.add_custom_node(
        result=bool(sch and sch.field and sch.field.strip()),
        id="FieldOfScholarshipIdentified",
        desc="The field or area of scholarly work is mentioned",
        parent=node,
        critical=False
    )


async def verify_conference_affiliation(evaluator: Evaluator, parent, person: PersonInfo):
    node = evaluator.add_parallel(
        id="AthleticConferenceAffiliation",
        desc="Verification that the current university is a member of a Power Five athletic conference",
        parent=parent,
        critical=True
    )

    # Reference URL for conference exists (prefer explicit conference_sources, fallback to current sources)
    conf_sources = []
    if person.current_position:
        conf_sources = (person.current_position.conference_sources or []) or person.current_position.sources
    ref_conf = evaluator.add_custom_node(
        result=bool(conf_sources),
        id="ReferenceURL_Conference",
        desc="Source URL confirming conference affiliation is provided",
        parent=node,
        critical=True
    )

    # Conference named & valid (Power Five)
    evaluator.add_custom_node(
        result=is_power5_name(person.current_position.conference if person.current_position else None),
        id="ConferenceNamedAndValid",
        desc="The athletic conference is explicitly named and is one of the Power Five (ACC, Big Ten, SEC, Big 12, or Pac-12)",
        parent=node,
        critical=True
    )

    # University membership correct (URL verification)
    univ_conf_leaf = evaluator.add_leaf(
        id="UniversityMembershipCorrect",
        desc="The current university's conference membership is correctly stated",
        parent=node,
        critical=True
    )
    claim_conf = f"{person.current_position.university if person.current_position else ''} is a member of the {normalize_conference(person.current_position.conference) if person.current_position else ''}."
    await evaluator.verify(
        claim=claim_conf,
        node=univ_conf_leaf,
        sources=conf_sources,
        additional_instruction="Verify conference membership via official conference, athletics, university sites, or reputable news.",
        extra_prerequisites=[ref_conf]
    )


async def verify_tenure_in_current_role(evaluator: Evaluator, parent, person: PersonInfo):
    node = evaluator.add_sequential(
        id="TenureInCurrentRole",
        desc="Verification that the individual has served in their current role for at least 10 consecutive years as of 2026",
        parent=parent,
        critical=True
    )

    # Start year info (parallel group)
    start_group = evaluator.add_parallel(
        id="StartYearInformation",
        desc="The year the individual began their current presidential/chancellor role is stated and verified",
        parent=node,
        critical=True
    )

    # Reference URL for start year (prefer explicit start_year_sources, fallback to current sources)
    start_sources = []
    if person.current_position:
        start_sources = (person.current_position.start_year_sources or []) or person.current_position.sources
    ref_start = evaluator.add_custom_node(
        result=bool(start_sources),
        id="ReferenceURL_StartYear",
        desc="Source URL confirming start date of current role is provided",
        parent=start_group,
        critical=True
    )

    # Start year accurate (URL verification)
    start_leaf = evaluator.add_leaf(
        id="StartYearAccurate",
        desc="The stated start year is correct and verified against official sources",
        parent=start_group,
        critical=True
    )
    claim_start = f"{person.full_name} began serving as {person.current_position.title if person.current_position else 'the current role'} at {person.current_position.university if person.current_position else ''} in {person.current_position.start_year if person.current_position else ''}."
    await evaluator.verify(
        claim=claim_start,
        node=start_leaf,
        sources=start_sources,
        additional_instruction="Verify the start year from official bios, press releases, or reputable news coverage. Allow that sources may say 'since 2012' or include month/day.",
        extra_prerequisites=[ref_start]
    )

    # Tenure requirement met (custom arithmetic check)
    yr = first_int_year(person.current_position.start_year if person.current_position else None)
    evaluator.add_custom_node(
        result=bool(yr is not None and yr <= 2016),
        id="TenureRequirementMet",
        desc="The individual has served for at least 10 consecutive years (started in 2016 or earlier) with continuous service",
        parent=node,
        critical=True
    )


async def verify_supporting_evidence(evaluator: Evaluator, parent, person: PersonInfo):
    node = evaluator.add_parallel(
        id="SupportingEvidence",
        desc="Quality and completeness of reference URLs and documentation",
        parent=parent,
        critical=False  # Adjusted to allow non-critical child SourcesAccessible
    )

    all_urls = aggregate_all_sources(person)

    # Official sources used (at least one official university source)
    evaluator.add_custom_node(
        result=any(is_official_university_url(u) for u in all_urls),
        id="OfficialSourcesUsed",
        desc="At least one URL from an official university source is provided",
        parent=node,
        critical=True
    )

    # Sources credible (all urls credible)
    evaluator.add_custom_node(
        result=all(is_credible_source(u) for u in all_urls) if all_urls else False,
        id="SourcesCredible",
        desc="All reference URLs are from credible sources (official university sites, reputable news organizations, academic databases)",
        parent=node,
        critical=True
    )

    # Sources accessible (non-critical). We weakly check at least one page loads by URL verification.
    # Since verifying all would be costly, we check representative accessibility: any provided URL returns content.
    accessible_leaf = evaluator.add_leaf(
        id="SourcesAccessible",
        desc="Provided URLs are valid and accessible",
        parent=node,
        critical=False
    )
    # Verify accessibility with multi-URL: treat as non-critical sanity check
    claim_accessible = "This webpage is accessible and loads content."
    await evaluator.verify(
        claim=claim_accessible,
        node=accessible_leaf,
        sources=all_urls[:8] if len(all_urls) > 8 else all_urls,  # cap for efficiency
        additional_instruction="Just confirm that the page loads and contains real content; no need to verify specific facts."
    )

    # Key facts supported (each major claim has at least one supporting URL)
    current_ok = bool(person.current_position and person.current_position.sources)
    dean_ok = bool(pick_dean_position(person.dean_positions) and pick_dean_position(person.dean_positions).sources)
    degree_ok = bool(person.degree and person.degree.sources)
    scholarship_ok = bool(person.scholarship and person.scholarship.sources)
    conference_ok = bool(person.current_position and ((person.current_position.conference_sources and len(person.current_position.conference_sources) > 0) or (person.current_position.sources and len(person.current_position.sources) > 0)))
    start_ok = bool(person.current_position and ((person.current_position.start_year_sources and len(person.current_position.start_year_sources) > 0) or (person.current_position.sources and len(person.current_position.sources) > 0)))
    evaluator.add_custom_node(
        result=all([current_ok, dean_ok, degree_ok, scholarship_ok, conference_ok, start_ok]),
        id="KeyFactsSupported",
        desc="Each major claim has at least one supporting reference URL",
        parent=node,
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
    # Initialize evaluator with sequential root (root is non-critical to allow partials where defined)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured info
    person = await evaluator.extract(
        prompt=prompt_extract_person_info(),
        template_class=PersonInfo,
        extraction_name="extracted_person_info"
    )

    # Build verification tree per rubric (sequential aggregation at root)
    await verify_person_identification(evaluator, root, person)
    await verify_previous_deanship(evaluator, root, person)
    await verify_doctoral_degree(evaluator, root, person)
    await verify_scholarly_work(evaluator, root, person)
    await verify_conference_affiliation(evaluator, root, person)
    await verify_tenure_in_current_role(evaluator, root, person)
    await verify_supporting_evidence(evaluator, root, person)

    # Custom info: list accepted Power Five conferences (for transparency)
    evaluator.add_custom_info(
        info={"accepted_power5": ["ACC", "Big Ten", "SEC", "Big 12", "Pac-12"], "tenure_cutoff_year": 2016},
        info_type="config",
        info_name="evaluation_parameters"
    )

    return evaluator.get_summary()