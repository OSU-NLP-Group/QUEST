import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "early_cs_faculty_r12"
TASK_DESCRIPTION = """Identify four early-career computer science faculty members currently employed at U.S. research universities (R1 or R2 Carnegie Classification) who each satisfy all of the following criteria:

Position Requirements:
- Holds a tenure-track Assistant Professor position
- Works in a Computer Science department or a closely related department (such as Electrical and Computer Engineering, Information Science, or Data Science)
- Started their faculty appointment between 2019 and 2024

Publication Requirements:
- Has at least 5 peer-reviewed publications since completing their PhD
- Has at least 2 first-author publications
- Has at least 1 publication at a top-tier conference, defined as either having CORE A* or A ranking, or an acceptance rate of 30% or lower

Collaboration Requirements:
- Has at least 1 multi-institutional publication where co-authors are affiliated with at least 2 different institutions

Research Area:
- Has a clearly identified primary research area in computer science

For each faculty member, provide:
1. Full name and current position title
2. Name and URL of their affiliated university
3. URL to their official university faculty profile page
4. Year they started their current faculty position
5. List of at least 5 publications with titles
6. For at least 2 publications: indicate that the faculty member is the first author
7. For at least 1 top-tier conference publication: provide the paper title, conference name, year, and a URL to the paper or conference proceedings, along with verification that the conference meets the top-tier criteria (CORE ranking or acceptance rate)
8. For at least 1 multi-institutional publication: provide the paper title, list of co-author affiliations showing at least 2 different institutions, and a URL to the paper
9. Primary research area or specialization
10. URL to their Google Scholar profile showing total citations and h-index
"""


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class PublicationEntry(BaseModel):
    title: Optional[str] = None
    venue: Optional[str] = None
    year: Optional[str] = None
    url: Optional[str] = None
    is_first_author: Optional[bool] = None


class TopTierInfo(BaseModel):
    title: Optional[str] = None
    conference_name: Optional[str] = None
    year: Optional[str] = None
    paper_url: Optional[str] = None
    criterion_url: Optional[str] = None  # URL verifying CORE A/A* or <=30% acceptance


class MultiInstitutionInfo(BaseModel):
    title: Optional[str] = None
    paper_url: Optional[str] = None
    coauthor_affiliations: List[str] = Field(default_factory=list)


class FacultyItem(BaseModel):
    full_name: Optional[str] = None
    position_title: Optional[str] = None
    department_name: Optional[str] = None
    department_url: Optional[str] = None
    university_name: Optional[str] = None
    university_url: Optional[str] = None
    r1r2_verification_url: Optional[str] = None
    profile_url: Optional[str] = None
    start_year: Optional[str] = None
    start_year_evidence_url: Optional[str] = None
    phd_completion_year: Optional[str] = None
    phd_evidence_url: Optional[str] = None
    publications: List[PublicationEntry] = Field(default_factory=list)
    first_author_publications: List[PublicationEntry] = Field(default_factory=list)
    top_tier: Optional[TopTierInfo] = None
    multi_institution: Optional[MultiInstitutionInfo] = None
    primary_research_area: Optional[str] = None
    scholar_url: Optional[str] = None


class FacultyExtraction(BaseModel):
    faculties: List[FacultyItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_faculties() -> str:
    return """
Extract up to FOUR (4) faculty members exactly as presented in the answer. For each faculty member, extract the following fields. If something is not present in the answer, set it to null (or an empty list for arrays). Do not invent any information. Extract URLs exactly as they appear (plain or markdown).

For each faculty member, provide an object with:
- full_name: string
- position_title: string (as written; e.g., "Assistant Professor")
- department_name: string (e.g., "Computer Science", "Electrical and Computer Engineering", "Informatics")
- department_url: string URL if provided, else null
- university_name: string (institution name)
- university_url: string URL if provided, else null
- r1r2_verification_url: string URL that shows the Carnegie classification (R1 or R2) OR a page that clearly states it (e.g., an official page or Wikipedia listing). If none provided, set null.
- profile_url: string URL to the official faculty profile page at the university
- start_year: string year they started current faculty appointment (e.g., "2019")
- start_year_evidence_url: string URL that mentions the start year (often the profile page or an announcement). If not clear, use the most relevant official page; else null.
- phd_completion_year: string year of PhD completion (e.g., "2020")
- phd_evidence_url: string URL that evidences the PhD completion year (profile, CV, or similar). If none, set null.

- publications: array of publication objects (include all that the answer lists up to at least 5). Each publication object fields:
  - title: string
  - venue: string (conference/journal name if provided)
  - year: string year (e.g., "2022")
  - url: string URL to the paper or official page
  - is_first_author: boolean (true if the faculty is the first author according to the answer; else false or null)

- first_author_publications: array of at least two publications (from the answer) where the faculty is first author. Each item uses the same fields as 'publications' above. If not available, return an empty array.

- top_tier: object with details for one (or more) top-tier publication (CORE A*/A or acceptance rate ≤ 30%):
  - title: string
  - conference_name: string
  - year: string
  - paper_url: string
  - criterion_url: string URL to a page that verifies CORE A*/A ranking or shows acceptance rate ≤ 30%

- multi_institution: object describing at least one multi-institutional paper:
  - title: string
  - paper_url: string
  - coauthor_affiliations: array of strings listing affiliations for co-authors (as shown in the answer). If not listed explicitly, leave empty.

- primary_research_area: string (e.g., "Computer Vision", "Databases", "Security", "NLP", "HCI")
- scholar_url: string URL to Google Scholar profile

Rules:
- Extract only what is in the answer. If multiple candidates are given, extract the first four.
- Preserve text as-is. For years, keep strings.
- For URLs, extract only valid URLs. If a URL is given in markdown, extract the actual link target.
"""


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def urls_list(*candidates: Optional[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for c in candidates:
        if non_empty(c):
            v = str(c).strip()
            if v not in seen:
                out.append(v)
                seen.add(v)
    return out


def parse_year(s: Optional[str]) -> Optional[int]:
    if not non_empty(s):
        return None
    m = re.search(r"(19|20)\d{2}", str(s))
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def count_post_phd_pubs(publications: List[PublicationEntry], phd_year: Optional[int]) -> int:
    if phd_year is None:
        return 0
    cnt = 0
    for p in publications:
        y = parse_year(p.year)
        if y is not None and y > phd_year and non_empty(p.title) and non_empty(p.url):
            cnt += 1
    return cnt


def count_first_author_pubs(faculty: FacultyItem) -> int:
    cnt = 0
    # Prefer explicit first_author_publications list
    for p in faculty.first_author_publications:
        if non_empty(p.title) and non_empty(p.url):
            cnt += 1
    # Also count flags from general publications (avoid double counting by title+url signature)
    seen = {(p.title or "", p.url or "") for p in faculty.first_author_publications if non_empty(p.title) and non_empty(p.url)}
    for p in faculty.publications:
        if p.is_first_author and non_empty(p.title) and non_empty(p.url):
            key = (p.title or "", p.url or "")
            if key not in seen:
                cnt += 1
                seen.add(key)
    return cnt


# -----------------------------------------------------------------------------
# Verification subroutines
# -----------------------------------------------------------------------------
async def verify_affiliation_and_role(evaluator: Evaluator, parent, faculty: FacultyItem, idx: int):
    prefix = f"F{idx}"
    node = evaluator.add_parallel(
        id=f"{prefix}_AffiliationAndRole",
        desc=f"Identity, role, institution, department, and start-year constraints for Faculty {idx}",
        parent=parent,
        critical=True,
    )

    # Full name
    evaluator.add_custom_node(
        result=non_empty(faculty.full_name),
        id=f"{prefix}_FullName",
        desc="Full name is provided",
        parent=node,
        critical=True,
    )

    # University name
    evaluator.add_custom_node(
        result=non_empty(faculty.university_name),
        id=f"{prefix}_UniversityName",
        desc="Affiliated university name is provided",
        parent=node,
        critical=True,
    )

    # University URL
    evaluator.add_custom_node(
        result=non_empty(faculty.university_url),
        id=f"{prefix}_UniversityURL",
        desc="Affiliated university URL is provided",
        parent=node,
        critical=True,
    )

    # Official profile URL
    evaluator.add_custom_node(
        result=non_empty(faculty.profile_url),
        id=f"{prefix}_ProfileURL",
        desc="Official university faculty profile page URL is provided",
        parent=node,
        critical=True,
    )

    # Department fit (CS or closely related)
    dept_fit_leaf = evaluator.add_leaf(
        id=f"{prefix}_DepartmentFit",
        desc="Department is CS or closely related (evidence via profile/department page)",
        parent=node,
        critical=True,
    )
    dept_sources = urls_list(faculty.profile_url, faculty.department_url)
    dept_claim = (
        "The faculty member's department is Computer Science or a closely related computing field, such as "
        "Electrical and Computer Engineering (ECE), Information Science, Informatics, Data Science, or a School/Department of Computing."
    )
    await evaluator.verify(
        claim=dept_claim,
        node=dept_fit_leaf,
        sources=dept_sources if dept_sources else None,
        additional_instruction="Accept departments explicitly in computing (Computer Science/Engineering, ECE, CSE, Informatics, Information/Computer/Computing). Reject purely non-computing departments.",
    )

    # Position is tenure-track Assistant Professor
    position_leaf = evaluator.add_leaf(
        id=f"{prefix}_Position",
        desc="Current position is tenure-track Assistant Professor (evidence via profile URL)",
        parent=node,
        critical=True,
    )
    pos_sources = urls_list(faculty.profile_url)
    pos_claim = (
        f"According to the official faculty profile page, {faculty.full_name or 'the faculty member'} holds a tenure-track Assistant Professor position."
    )
    await evaluator.verify(
        claim=pos_claim,
        node=position_leaf,
        sources=pos_sources if pos_sources else None,
        additional_instruction="Confirm that the title is 'Assistant Professor' and on a tenure-track (not teaching/clinical/research professor, lecturer, or adjunct). Titles like 'Assistant Professor' in research/teaching tracks are not tenure-track unless explicitly stated.",
    )

    # R1/R2 verification
    r1r2_leaf = evaluator.add_leaf(
        id=f"{prefix}_R1R2Verification",
        desc="Evidence/URL verifies the university is U.S.-based and Carnegie R1 or R2",
        parent=node,
        critical=True,
    )
    r1r2_sources = urls_list(faculty.r1r2_verification_url, faculty.university_url)
    r1r2_claim = (
        f"The affiliated university {faculty.university_name or 'the institution'} is a U.S. university and classified as Carnegie R1 (Very High Research Activity) or R2 (High Research Activity)."
    )
    await evaluator.verify(
        claim=r1r2_claim,
        node=r1r2_leaf,
        sources=r1r2_sources if r1r2_sources else None,
        additional_instruction="Prefer official Carnegie classifications or reliable listings (e.g., Carnegie website, credible lists/wikis). The page should indicate 'R1' or 'R2' and that it's in the United States.",
    )

    # Start year (2019–2024 inclusive), verify via URL
    start_leaf = evaluator.add_leaf(
        id=f"{prefix}_StartYear",
        desc="Start year for current faculty appointment is provided and is 2019–2024 (evidence via URL)",
        parent=node,
        critical=True,
    )
    start_sources = urls_list(faculty.start_year_evidence_url, faculty.profile_url)
    start_claim = (
        f"According to the provided source, {faculty.full_name or 'the faculty member'} started their current faculty appointment in {faculty.start_year or '[missing]'}, "
        f"which is between 2019 and 2024 inclusive."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=start_sources if start_sources else None,
        additional_instruction="Verify the exact start year stated on the page and confirm it is within 2019–2024 inclusive.",
    )


async def verify_publication_requirements(evaluator: Evaluator, parent, faculty: FacultyItem, idx: int):
    prefix = f"F{idx}"
    node = evaluator.add_parallel(
        id=f"{prefix}_PublicationRequirements",
        desc=f"Publication constraints for Faculty {idx}, with publication links for verification",
        parent=parent,
        critical=True,
    )

    # PhD completion evidence (year + URL)
    phd_year_present = non_empty(faculty.phd_completion_year)
    phd_sources = urls_list(faculty.phd_evidence_url, faculty.profile_url)
    if phd_year_present and phd_sources:
        phd_leaf = evaluator.add_leaf(
            id=f"{prefix}_PhDCompletionEvidence",
            desc="Evidence/URL provides PhD completion year (needed to verify post-PhD publications)",
            parent=node,
            critical=True,
        )
        phd_claim = (
            f"The page indicates that {faculty.full_name or 'the faculty member'} completed their PhD in {faculty.phd_completion_year}."
        )
        await evaluator.verify(
            claim=phd_claim,
            node=phd_leaf,
            sources=phd_sources,
            additional_instruction="Look for sections listing education (e.g., 'PhD, 2020') on the profile/CV or similar official pages.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{prefix}_PhDCompletionEvidence",
            desc="Evidence/URL provides PhD completion year (needed to verify post-PhD publications)",
            parent=node,
            critical=True,
        )

    # ≥ 5 peer-reviewed publications since PhD completion
    phd_year_int = parse_year(faculty.phd_completion_year)
    post_phd_count = count_post_phd_pubs(faculty.publications, phd_year_int)
    has_5_post_phd = post_phd_count >= 5
    # Also ensure that at least 5 publications with title and url are present at all
    at_least_5_listed = sum(1 for p in faculty.publications if non_empty(p.title) and non_empty(p.url)) >= 5
    evaluator.add_custom_node(
        result=has_5_post_phd and at_least_5_listed and phd_year_int is not None,
        id=f"{prefix}_FivePostPhDPeerReviewedPubsWithLinks",
        desc="Provide ≥5 peer-reviewed publications since PhD completion; each entry includes at least title + venue/year + URL sufficient for verification",
        parent=node,
        critical=True,
    )

    # ≥ 2 first-author publications
    fa_count = count_first_author_pubs(faculty)
    evaluator.add_custom_node(
        result=fa_count >= 2,
        id=f"{prefix}_TwoFirstAuthorPubsVerified",
        desc="Identify ≥2 first-author publications with URLs showing author order (or equivalent verifiable evidence)",
        parent=node,
        critical=True,
    )

    # Top-tier conference publication details
    tt_node = evaluator.add_parallel(
        id=f"{prefix}_TopTierConferencePubVerified",
        desc="Provide ≥1 top-tier conference publication and verify top-tier status (CORE A*/A or acceptance rate ≤30%)",
        parent=node,
        critical=True,
    )

    # Existence leaves for top-tier subfields
    evaluator.add_custom_node(
        result=(faculty.top_tier is not None and non_empty(faculty.top_tier.title)),
        id=f"{prefix}_TopTier_PaperTitle",
        desc="Top-tier paper title is provided",
        parent=tt_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(faculty.top_tier is not None and non_empty(faculty.top_tier.conference_name)),
        id=f"{prefix}_TopTier_ConferenceName",
        desc="Conference name is provided",
        parent=tt_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(faculty.top_tier is not None and non_empty(faculty.top_tier.year)),
        id=f"{prefix}_TopTier_Year",
        desc="Publication/conference year is provided",
        parent=tt_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(faculty.top_tier is not None and non_empty(faculty.top_tier.paper_url)),
        id=f"{prefix}_TopTier_PaperOrProceedingsURL",
        desc="URL to paper or official proceedings is provided",
        parent=tt_node,
        critical=True,
    )

    # Verify the criterion (CORE A*/A or acceptance rate ≤30%)
    tt_crit_leaf = evaluator.add_leaf(
        id=f"{prefix}_TopTier_CriterionVerificationURL",
        desc="URL/evidence verifies CORE A*/A ranking OR acceptance rate ≤30%",
        parent=tt_node,
        critical=True,
    )
    if faculty.top_tier and non_empty(faculty.top_tier.criterion_url):
        tt_crit_claim = (
            f"The conference {faculty.top_tier.conference_name or 'the conference'} is a top-tier venue (CORE A* or A) "
            f"or has an acceptance rate of 30% or lower."
        )
        await evaluator.verify(
            claim=tt_crit_claim,
            node=tt_crit_leaf,
            sources=faculty.top_tier.criterion_url,
            additional_instruction="Accept if the page clearly shows CORE rank A* or A (any recent CORE list) or reports an acceptance rate ≤ 30%.",
        )
    else:
        # No URL to verify criterion -> fail this leaf
        tt_crit_leaf.score = 0.0
        tt_crit_leaf.status = "failed"


async def verify_collaboration(evaluator: Evaluator, parent, faculty: FacultyItem, idx: int):
    prefix = f"F{idx}"
    node = evaluator.add_parallel(
        id=f"{prefix}_CollaborationRequirement",
        desc=f"Multi-institutional publication requirement for Faculty {idx}",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(faculty.multi_institution is not None and non_empty(faculty.multi_institution.title)),
        id=f"{prefix}_MultiInst_PaperTitle",
        desc="Multi-institution paper title is provided",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(faculty.multi_institution is not None and non_empty(faculty.multi_institution.paper_url)),
        id=f"{prefix}_MultiInst_PaperURL",
        desc="URL to the paper (showing authors/affiliations) is provided",
        parent=node,
        critical=True,
    )

    # Verify at least 2 distinct institutions in affiliations on provided URL
    aff_leaf = evaluator.add_leaf(
        id=f"{prefix}_MultiInst_AffiliationsEvidence",
        desc="Co-author affiliations show ≥2 distinct institutions (verifiable from provided URL)",
        parent=node,
        critical=True,
    )
    if faculty.multi_institution and non_empty(faculty.multi_institution.paper_url):
        aff_claim = (
            "This paper's author information shows co-authors affiliated with at least two different institutions "
            "(multi-institutional collaboration)."
        )
        await evaluator.verify(
            claim=aff_claim,
            node=aff_leaf,
            sources=faculty.multi_institution.paper_url,
            additional_instruction="Look for author affiliation lines or institution footnotes/metadata indicating multiple distinct institutions.",
        )
    else:
        aff_leaf.score = 0.0
        aff_leaf.status = "failed"


async def verify_research_area_and_scholar(evaluator: Evaluator, parent, faculty: FacultyItem, idx: int):
    prefix = f"F{idx}"
    node = evaluator.add_parallel(
        id=f"{prefix}_ResearchAreaAndScholar",
        desc=f"Research area and Google Scholar requirements for Faculty {idx}",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=non_empty(faculty.primary_research_area),
        id=f"{prefix}_PrimaryResearchArea",
        desc="Primary CS research area/specialization is provided",
        parent=node,
        critical=True,
    )

    scholar_leaf = evaluator.add_leaf(
        id=f"{prefix}_GoogleScholarEvidence",
        desc="Google Scholar profile URL is provided and shows total citations and h-index",
        parent=node,
        critical=True,
    )
    if non_empty(faculty.scholar_url):
        scholar_claim = "This Google Scholar profile page shows both total citations and h-index metrics."
        await evaluator.verify(
            claim=scholar_claim,
            node=scholar_leaf,
            sources=faculty.scholar_url,
            additional_instruction="Verify that the page includes numerical fields labeled 'Citations' (total) and 'h-index'.",
        )
    else:
        scholar_leaf.score = 0.0
        scholar_leaf.status = "failed"


async def verify_faculty_bundle(evaluator: Evaluator, parent, faculty: FacultyItem, idx: int):
    """Builds the Faculty_i subtree and runs all required checks."""
    fac_node = evaluator.add_parallel(
        id=f"Faculty_{idx}",
        desc=f"Faculty member #{idx} (qualifying candidate with required evidence/URLs)",
        parent=parent,
        critical=False,
    )
    await verify_affiliation_and_role(evaluator, fac_node, faculty, idx)
    await verify_publication_requirements(evaluator, fac_node, faculty, idx)
    await verify_collaboration(evaluator, fac_node, faculty, idx)
    await verify_research_area_and_scholar(evaluator, fac_node, faculty, idx)


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    Entry point for the early-career CS faculty (R1/R2) evaluation.
    """
    evaluator = Evaluator()
    # Important: set root as non-critical to allow partial credit across 4 faculty
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_faculties(),
        template_class=FacultyExtraction,
        extraction_name="faculty_extraction",
    )

    # Normalize to exactly 4 faculty items (pad with empty if fewer; truncate if more)
    faculties: List[FacultyItem] = list(extracted.faculties or [])
    if len(faculties) < 4:
        faculties = faculties + [FacultyItem() for _ in range(4 - len(faculties))]
    if len(faculties) > 4:
        faculties = faculties[:4]

    # Add some custom info diagnostics
    evaluator.add_custom_info(
        info={"num_faculty_extracted": len(extracted.faculties or []), "num_faculty_evaluated": 4},
        info_type="diagnostics",
        info_name="extraction_stats",
    )

    # Build the verification tree for each faculty
    for i in range(4):
        await verify_faculty_bundle(evaluator, root, faculties[i], i + 1)

    # Return the evaluation summary
    return evaluator.get_summary()