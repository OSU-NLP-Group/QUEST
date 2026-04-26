import asyncio
import logging
import re
from typing import Any, Optional, List, Dict, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "odu_3_faculty_arx2020_2025"
TASK_DESCRIPTION = (
    "Identify three faculty members currently at Old Dominion University who meet ALL of the following criteria: "
    "(1) Conduct research in one of ODU's four nationally recognized research strength areas (coastal resilience, "
    "modeling and simulation, bioelectrics, or cybersecurity), (2) Hold a graduate degree (Master's or PhD) in an "
    "engineering discipline that matches one of the Artemis II crew members' educational backgrounds (electrical "
    "engineering, general engineering, systems engineering, or computer engineering), and (3) Have published at least "
    "one peer-reviewed research paper between 2020 and 2025 (inclusive). For each faculty member, provide their name "
    "and current title/position at ODU, their research area (one of the four specified areas), their graduate degree "
    "information (degree type, field, and granting institution), one peer-reviewed publication from 2020-2025 "
    "(including title, year, and venue), and supporting URL references for verification."
)

# Allowed research areas and disciplines
ALLOWED_RESEARCH_AREAS_CANONICAL = [
    "coastal resilience",
    "modeling and simulation",
    "bioelectrics",
    "cybersecurity",
]

ALLOWED_DISCIPLINES = [
    "electrical engineering",
    "general engineering",
    "systems engineering",
    "computer engineering",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DegreeInfo(BaseModel):
    degree_type: Optional[str] = None  # e.g., "MS", "M.S.", "Master of Science", "PhD", "Doctor of Philosophy"
    field: Optional[str] = None        # e.g., "Electrical Engineering"
    institution: Optional[str] = None  # e.g., "North Carolina State University"


class PublicationInfo(BaseModel):
    title: Optional[str] = None
    year: Optional[str] = None  # keep as string to be robust to formats like "2021" or "2021 (online first)"
    venue: Optional[str] = None  # journal or conference
    urls: List[str] = Field(default_factory=list)


class FacultyMember(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    affiliation_urls: List[str] = Field(default_factory=list)

    research_area: Optional[str] = None                # expected to be one of the allowed areas (or a close synonym)
    research_urls: List[str] = Field(default_factory=list)

    degree: DegreeInfo = DegreeInfo()
    education_urls: List[str] = Field(default_factory=list)

    publication: PublicationInfo = PublicationInfo()


class ODUFacultyExtraction(BaseModel):
    faculty: List[FacultyMember] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_faculty() -> str:
    return """
    Extract up to the first three Old Dominion University (ODU) faculty members mentioned in the answer who are proposed to meet the specified criteria.
    For each faculty member, extract the following fields exactly as stated in the answer (use null or empty list if missing):

    - name: the full name of the faculty member
    - title: their current title/position at ODU
    - affiliation_urls: an array of URLs that confirm ODU affiliation (department page, profile page, etc.)
    - research_area: the stated research area (ideally one of: coastal resilience, modeling and simulation, bioelectrics, or cybersecurity)
    - research_urls: an array of URLs that support the stated research area (lab page, research profile, project page, etc.)
    - degree: an object with:
        - degree_type: the type of the graduate degree (e.g., MS, M.S., Master of Science, PhD, Doctor of Philosophy)
        - field: the degree field (e.g., Electrical Engineering, Systems Engineering)
        - institution: the granting institution (e.g., North Carolina State University)
    - education_urls: an array of URLs supporting the degree information
    - publication: an object with:
        - title: title of one peer-reviewed publication (preferably from 2020-2025; if multiple are given, choose one)
        - year: the publication year (preferably a value 2020-2025)
        - venue: the journal or conference venue name
        - urls: an array of URLs to the publication record or venue page (DOI, publisher page, Google Scholar, etc.)

    Rules:
    - Only extract what is explicitly present in the answer; do not invent details.
    - If the answer provides more than three faculty, only include the first three entries.
    - If any field is missing, set it to null or an empty array as appropriate.
    - Preserve raw text for fields (e.g., do not normalize or paraphrase).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def is_graduate_degree(degree_type: Optional[str]) -> bool:
    """Check if degree_type indicates Master's or PhD (graduate degree)."""
    s = _norm(degree_type)
    if not s:
        return False
    patterns = [
        r"\bms\b", r"\bm\.s\.\b", r"\bmaster\b", r"\bmasters\b", r"\bmaster of science\b",
        r"\bphd\b", r"\bph\.d\.\b", r"\bdoctor of philosophy\b", r"\bdoctoral\b", r"\bdoctorate\b",
        r"\bmeng\b", r"\bm\.eng\.\b", r"\bmaster of engineering\b",
    ]
    return any(re.search(p, s) for p in patterns)


def matches_allowed_discipline(field: Optional[str]) -> bool:
    """Check whether the degree field matches the allowed engineering disciplines."""
    s = _norm(field)
    if not s or "engineering" not in s:
        return False

    # Match core keywords
    if "electrical" in s:
        return True
    if "systems" in s:
        # Ensure it's not purely 'information systems' without engineering
        return "engineering" in s
    if "computer engineering" in s:
        return True
    if "general engineering" in s:
        return True

    # Accept combined fields like "electrical and computer engineering"
    if "electrical and computer engineering" in s or "electrical & computer engineering" in s:
        return True

    return False


def normalize_area(area: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Try to map the provided research area string to one of the canonical areas."""
    s = _norm(area)
    if not s:
        return False, None

    # Synonyms / fuzzy matches
    if "coastal" in s and ("resilience" in s or "resiliency" in s or "engineering" in s):
        return True, "coastal resilience"
    if ("modeling" in s or "modelling" in s) and "simulation" in s:
        return True, "modeling and simulation"
    if "simulation" in s and "model" in s:
        return True, "modeling and simulation"
    if "bioelectric" in s or "bioelectrics" in s:
        return True, "bioelectrics"
    if "cybersecurity" in s or ("cyber" in s and "security" in s):
        return True, "cybersecurity"

    # Exact fallback
    if s in ALLOWED_RESEARCH_AREAS_CANONICAL:
        return True, s

    return False, None


def year_in_range(year_str: Optional[str], min_year: int = 2020, max_year: int = 2025) -> bool:
    """Check whether a year string contains a valid year in the inclusive range."""
    if not year_str:
        return False
    m = re.search(r"(20\d{2})", year_str)
    if not m:
        return False
    try:
        y = int(m.group(1))
        return min_year <= y <= max_year
    except Exception:
        return False


def non_empty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any((u or "").strip() for u in urls)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_faculty(
    evaluator: Evaluator,
    parent_node,
    faculty: FacultyMember,
    idx: int,
) -> None:
    """
    Build verification sub-tree for a single faculty member and run verifications.
    """

    # Top-level node for this faculty member (non-critical to allow partial credit across the three)
    fac_node = evaluator.add_parallel(
        id=f"faculty_member_{idx + 1}",
        desc=f"{['First', 'Second', 'Third'][idx] if idx < 3 else f'#{idx+1}'} qualifying faculty member meeting all criteria",
        parent=parent_node,
        critical=False
    )

    # ------------------ Basic Affiliation ------------------ #
    basic_aff_node = evaluator.add_parallel(
        id=f"fm{idx + 1}_basic_affiliation",
        desc="Faculty member is affiliated with Old Dominion University and basic information is provided",
        parent=fac_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(faculty.name and faculty.name.strip()),
        id=f"fm{idx + 1}_name_provided",
        desc="Faculty member's name is provided",
        parent=basic_aff_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(faculty.title and faculty.title.strip()),
        id=f"fm{idx + 1}_title_provided",
        desc="Faculty member's current title/position at ODU is provided",
        parent=basic_aff_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_urls(faculty.affiliation_urls),
        id=f"fm{idx + 1}_affiliation_url",
        desc="Provide URL reference confirming ODU affiliation",
        parent=basic_aff_node,
        critical=True
    )

    current_employment_node = evaluator.add_leaf(
        id=f"fm{idx + 1}_current_employment",
        desc="Verify current employment or research appointment at ODU",
        parent=basic_aff_node,
        critical=True
    )
    claim_aff = f"{faculty.name or 'The faculty member'} currently holds the position '{faculty.title or ''}' at Old Dominion University."
    await evaluator.verify(
        claim=claim_aff,
        node=current_employment_node,
        sources=faculty.affiliation_urls,
        additional_instruction="Use the provided ODU affiliation URLs to confirm that the person is currently employed or has an active appointment at ODU. Allow reasonable title wording variants (e.g., Assistant/Associate Professor, Professor, Research Professor, etc.)."
    )

    # ------------------ Research Area Verification ------------------ #
    research_node = evaluator.add_parallel(
        id=f"fm{idx + 1}_research_area_verification",
        desc="Faculty member's research aligns with one of ODU's four nationally recognized research strength areas",
        parent=fac_node,
        critical=True
    )

    match_ok, canonical_area = normalize_area(faculty.research_area)
    evaluator.add_custom_node(
        result=match_ok,
        id=f"fm{idx + 1}_research_area_match",
        desc="Research area is one of: coastal resilience, modeling and simulation, bioelectrics, or cybersecurity",
        parent=research_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_urls(faculty.research_urls),
        id=f"fm{idx + 1}_research_area_url",
        desc="Provide URL reference confirming research area",
        parent=research_node,
        critical=True
    )

    research_evidence_node = evaluator.add_leaf(
        id=f"fm{idx + 1}_research_evidence",
        desc="Evidence of active research in the specified area",
        parent=research_node,
        critical=True
    )
    claim_research = f"{faculty.name or 'The faculty member'} conducts active research in {canonical_area or faculty.research_area or 'the specified area'}."
    combined_research_sources = (faculty.research_urls or []) + (faculty.affiliation_urls or [])
    await evaluator.verify(
        claim=claim_research,
        node=research_evidence_node,
        sources=combined_research_sources,
        additional_instruction="Confirm from the URLs that this faculty member's research explicitly aligns with the specified area (coastal resilience, modeling and simulation, bioelectrics, or cybersecurity). Consider lab pages, research profiles, and project pages as acceptable evidence."
    )

    # ------------------ Educational Background ------------------ #
    edu_node = evaluator.add_parallel(
        id=f"fm{idx + 1}_educational_background",
        desc="Faculty member holds graduate degree in qualifying engineering discipline",
        parent=fac_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_graduate_degree(faculty.degree.degree_type),
        id=f"fm{idx + 1}_degree_level",
        desc="Holds Master's degree or PhD",
        parent=edu_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=matches_allowed_discipline(faculty.degree.field),
        id=f"fm{idx + 1}_engineering_discipline",
        desc="Degree is in electrical engineering, general engineering, systems engineering, or computer engineering",
        parent=edu_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_urls(faculty.education_urls),
        id=f"fm{idx + 1}_education_url",
        desc="Provide URL reference confirming educational background",
        parent=edu_node,
        critical=True
    )

    degree_verify_node = evaluator.add_leaf(
        id=f"fm{idx + 1}_degree_verification",
        desc="Provide degree information including field and institution",
        parent=edu_node,
        critical=True
    )
    claim_degree = (
        f"{faculty.name or 'The faculty member'} holds a graduate degree "
        f"('{faculty.degree.degree_type or ''}') in '{faculty.degree.field or ''}' "
        f"from '{faculty.degree.institution or ''}'."
    )
    await evaluator.verify(
        claim=claim_degree,
        node=degree_verify_node,
        sources=faculty.education_urls,
        additional_instruction="Verify the degree level (Master's or PhD), the engineering field, and the granting institution from the provided education URLs."
    )

    # ------------------ Publication Record ------------------ #
    pub_node = evaluator.add_parallel(
        id=f"fm{idx + 1}_publication_record",
        desc="Faculty member has published peer-reviewed research between 2020-2025",
        parent=fac_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(faculty.publication.title and year_in_range(faculty.publication.year)),
        id=f"fm{idx + 1}_publication_existence",
        desc="At least one peer-reviewed publication exists in the specified timeframe",
        parent=pub_node,
        critical=True
    )

    pub_details_node = evaluator.add_parallel(
        id=f"fm{idx + 1}_publication_details",
        desc="Publication information provided",
        parent=pub_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(faculty.publication.title and faculty.publication.title.strip()),
        id=f"fm{idx + 1}_pub_title_provided",
        desc="Publication title provided",
        parent=pub_details_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=year_in_range(faculty.publication.year),
        id=f"fm{idx + 1}_pub_year_provided",
        desc="Publication year (2020-2025) provided",
        parent=pub_details_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(faculty.publication.venue and faculty.publication.venue.strip()),
        id=f"fm{idx + 1}_pub_venue_provided",
        desc="Journal or conference name provided",
        parent=pub_details_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_urls(faculty.publication.urls),
        id=f"fm{idx + 1}_publication_url",
        desc="Provide URL reference to publication or publication record",
        parent=pub_details_node,
        critical=True
    )

    peer_review_node = evaluator.add_leaf(
        id=f"fm{idx + 1}_peer_review_status",
        desc="Verification that publication is peer-reviewed",
        parent=pub_details_node,
        critical=True
    )
    claim_peer = (
        f"The publication '{faculty.publication.title or ''}' ({faculty.publication.year or ''}) at "
        f"'{faculty.publication.venue or ''}' is peer-reviewed."
    )
    await evaluator.verify(
        claim=claim_peer,
        node=peer_review_node,
        sources=faculty.publication.urls,
        additional_instruction="Use the publication record/venue pages to determine whether the venue is a peer-reviewed journal or a peer-reviewed scholarly conference. Recognized ACM/IEEE/Elsevier/Springer journals and major CS/engineering conferences are peer-reviewed."
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
    Evaluate an answer for the ODU faculty verification task.
    """
    # Initialize evaluator: root should be non-critical to allow partial credit across members
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # independent verification of each faculty member
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
    # Force root to be non-critical per framework consistency (critical parents must have all-critical children)
    root.critical = False

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_faculty(),
        template_class=ODUFacultyExtraction,
        extraction_name="odu_faculty_candidates"
    )

    # Select up to first three faculty; pad if fewer
    faculty_list: List[FacultyMember] = list(extraction.faculty[:3])
    while len(faculty_list) < 3:
        faculty_list.append(FacultyMember())

    # Add ground truth info (allowed areas and disciplines as context)
    evaluator.add_ground_truth({
        "allowed_research_areas": ALLOWED_RESEARCH_AREAS_CANONICAL,
        "allowed_disciplines": ALLOWED_DISCIPLINES,
        "timeframe_years_inclusive": [2020, 2025],
        "artemis_ii_degrees_reference": {
            "Christina Koch": "MS Electrical Engineering (NCSU)",
            "Victor Glover": "MS General Engineering",
            "Reid Wiseman": "MS Systems Engineering (Johns Hopkins University)"
        }
    }, gt_type="task_constraints")

    # Build verification subtrees for three faculty members
    for i, fac in enumerate(faculty_list):
        await verify_faculty(evaluator, root, fac, i)

    # Return evaluation summary
    return evaluator.get_summary()