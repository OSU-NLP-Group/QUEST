import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "new_cs_assistant_prof_2024_2025"
TASK_DESCRIPTION = """
Identify four assistant professors in Computer Science who were newly hired in the 2024-2025 academic year (starting Fall 2024 or later) at top-tier U.S. research universities (those appearing in major rankings such as CSRankings.org, US News Computer Science rankings, or QS World Rankings). These faculty members must specialize in artificial intelligence subfields: specifically machine learning, natural language processing, or computer vision.

For each of the four faculty members, provide the following information:

1. Full name and current position: The faculty member's complete name, official title (which must be Assistant Professor on the tenure track, not visiting, adjunct, or teaching-focused positions), and the name of their current institution.
2. PhD-granting institution: The university where the faculty member earned their doctoral degree, as stated in their official biography or profile.
3. Primary research area(s): The faculty member's main research focus in AI (machine learning, natural language processing, or computer vision) as explicitly listed on their university faculty profile.
4. Faculty profile URL: A direct link to the faculty member's official profile page on their university's Computer Science department website that confirms their appointment.
5. Google Scholar profile URL: A direct link to the faculty member's Google Scholar profile.
6. Recent publication: The title and publication venue (conference or journal) of at least one paper authored by this faculty member that was published after 2020. Include a link to this publication on Google Scholar or DBLP.

All information must be verifiable through official university websites, Google Scholar, or DBLP. Do not include faculty members who started before Fall 2024, or who hold ranks other than Assistant Professor on the tenure track.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PublicationInfo(BaseModel):
    title: Optional[str] = None
    venue: Optional[str] = None
    year: Optional[str] = None  # keep as string to be robust to formats
    urls: List[str] = Field(default_factory=list)  # Google Scholar or DBLP links


class FacultyMember(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None  # official title as stated
    institution: Optional[str] = None
    cs_department_name: Optional[str] = None  # e.g., "Department of Computer Science", "School of Computing"
    faculty_profile_url: Optional[str] = None  # official CS department profile page
    ranking_urls: List[str] = Field(default_factory=list)  # links to CSRankings/US News/QS showing inclusion
    research_areas: List[str] = Field(default_factory=list)  # list of areas as presented on profile
    phd_institution: Optional[str] = None
    phd_evidence_urls: List[str] = Field(default_factory=list)  # official profile or bio confirming PhD institution
    google_scholar_url: Optional[str] = None
    start_date_text: Optional[str] = None  # e.g., "Joined Fall 2024", "Started August 2024", "2025"
    start_evidence_urls: List[str] = Field(default_factory=list)  # official page(s) confirming start date
    publication: Optional[PublicationInfo] = None


class FacultyExtraction(BaseModel):
    members: List[FacultyMember] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_faculty() -> str:
    return """
    Extract up to four distinct faculty members satisfying the task requirements from the provided answer.
    Return a JSON object with a top-level 'members' array. For each member, extract:

    1. name: The faculty member's full name.
    2. title: The official academic title as stated (e.g., "Assistant Professor", avoid visiting/adjunct/teaching-focused positions).
    3. institution: The current university.
    4. cs_department_name: The name of the Computer Science department or school (e.g., "Department of Computer Science", "School of Computing"). If not explicitly stated, return null.
    5. faculty_profile_url: A direct URL to the official CS department faculty profile page that confirms the appointment.
    6. ranking_urls: An array of URLs (CSRankings.org, US News Computer Science, QS rankings) indicating the institution appears in a major CS ranking. If none are present in the answer, return an empty array.
    7. research_areas: An array of AI-related research areas explicitly listed on the official faculty profile (e.g., "machine learning", "natural language processing", "computer vision", "deep learning"). If not provided, return an empty array.
    8. phd_institution: The PhD-granting institution as stated on an official profile or biography page. If missing, return null.
    9. phd_evidence_urls: URLs to official pages confirming the PhD institution (e.g., the faculty profile or university bio page). Return an empty array if not present.
    10. google_scholar_url: A direct URL to the person's Google Scholar profile. If missing, return null.
    11. start_date_text: The exact text indicating the start date/term (e.g., "Joined Fall 2024", "Started in 2025", etc.). If missing, return null.
    12. start_evidence_urls: URLs to official pages confirming the start date or announcement (e.g., department news, hiring announcement). Return an empty array if not present.
    13. publication: An object describing a recent publication (post-2020) authored by the faculty member:
        - title: Paper title.
        - venue: Publication venue (conference/journal).
        - year: Year of publication as a string (e.g., "2023").
        - urls: Array of URLs that prove the publication (Google Scholar or DBLP). If none, return an empty array.

    Rules:
    - Only include the first four distinct faculty in the final 'members' array even if more are present.
    - Do not invent information. If a field is missing, set it to null or an empty array as appropriate.
    - For URLs, only include valid URLs explicitly given in the answer text (plain or markdown).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def first_four_members(members: List[FacultyMember]) -> List[FacultyMember]:
    return members[:4]


def are_four_distinct(members: List[FacultyMember]) -> bool:
    """Check exactly 4 members exist and they are distinct by (name, institution)."""
    if len(members) != 4:
        return False
    keys = []
    for m in members:
        nm = (m.name or "").strip().lower()
        inst = (m.institution or "").strip().lower()
        keys.append(f"{nm}::{inst}")
    return len(set(keys)) == 4 and all(k != "::" for k in keys)


# --------------------------------------------------------------------------- #
# Verification for a single faculty member                                    #
# --------------------------------------------------------------------------- #
async def verify_faculty_member(
    evaluator: Evaluator,
    parent_node,
    member: FacultyMember,
    idx: int,
) -> None:
    """
    Build verification nodes for one faculty member and run verifications according to rubric.
    """
    # Create a parallel aggregation node for this faculty member (non-critical to allow partial credit)
    fm_node = evaluator.add_parallel(
        id=f"faculty_member_{idx+1}",
        desc=f"Faculty member #{idx+1} (independently evaluated against all constraints and required fields)",
        parent=parent_node,
        critical=False
    )

    # Existence checks as critical custom nodes (gate subsequent verifications)
    identity_exists = evaluator.add_custom_node(
        result=bool(member.name and member.institution),
        id=f"fm_{idx+1}_identity_and_institution",
        desc="Provide full name and current institution.",
        parent=fm_node,
        critical=True
    )

    profile_exists = evaluator.add_custom_node(
        result=bool(member.faculty_profile_url),
        id=f"fm_{idx+1}_faculty_profile_url_exists",
        desc="Faculty profile URL is provided.",
        parent=fm_node,
        critical=True
    )

    scholar_exists = evaluator.add_custom_node(
        result=bool(member.google_scholar_url),
        id=f"fm_{idx+1}_gs_url_exists",
        desc="Google Scholar profile URL is provided.",
        parent=fm_node,
        critical=True
    )

    pub_exists = evaluator.add_custom_node(
        result=bool(member.publication and member.publication.title and member.publication.venue and member.publication.urls),
        id=f"fm_{idx+1}_publication_provided",
        desc="Recent publication information (title, venue, link) is provided.",
        parent=fm_node,
        critical=True
    )

    # CS department appointment verification
    cs_dept_leaf = evaluator.add_leaf(
        id=f"fm_{idx+1}_cs_department_appointment",
        desc="The appointment is in a Computer Science department (as evidenced on official university/CS department pages).",
        parent=fm_node,
        critical=True
    )
    cs_claim = (
        f"This page is an official Computer Science (or School of Computing) department profile page for {member.name} at {member.institution}."
    )
    await evaluator.verify(
        claim=cs_claim,
        node=cs_dept_leaf,
        sources=member.faculty_profile_url,
        additional_instruction="Confirm the page is on an official CS department or School of Computing site, and shows the person as faculty in CS. Look for 'Computer Science', 'School of Computing', or similar department identifiers on the page."
    )

    # Rank and track verification (Assistant Professor, tenure-track)
    rank_leaf = evaluator.add_leaf(
        id=f"fm_{idx+1}_rank_and_track",
        desc="Official title is Assistant Professor on the tenure track (not visiting/adjunct/teaching-focused), supported by an official university/department page.",
        parent=fm_node,
        critical=True
    )
    rank_claim = (
        f"The official title for {member.name} on this page indicates 'Assistant Professor' and is a tenure-track appointment (i.e., not 'Visiting', 'Adjunct', 'Teaching', 'Clinical', or 'Research Assistant Professor')."
    )
    await evaluator.verify(
        claim=rank_claim,
        node=rank_leaf,
        sources=member.faculty_profile_url,
        additional_instruction="Pass if the page title uses 'Assistant Professor' without qualifiers such as 'Visiting', 'Adjunct', 'Teaching', 'Clinical', or 'Research Assistant Professor'. Treat plain 'Assistant Professor' as tenure-track by default per typical US academic norms."
    )

    # Start date verification (Fall 2024 or later)
    start_leaf = evaluator.add_leaf(
        id=f"fm_{idx+1}_start_date_2024_2025",
        desc="Confirmed as a newly hired faculty member starting Fall 2024 or later (within the 2024–2025 academic year), supported by an official university/department page.",
        parent=fm_node,
        critical=True
    )
    start_sources = member.start_evidence_urls if member.start_evidence_urls else ([member.faculty_profile_url] if member.faculty_profile_url else [])
    start_claim = (
        f"{member.name} started in Fall 2024 or later at {member.institution}. Accept phrasing indicating joining/starting in late 2024 (e.g., August/September 2024) or any time in 2025."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=start_sources,
        additional_instruction="Look for text such as 'Joined Fall 2024', 'Started in 2025', 'Since August/September 2024', '2024-2025 academic year'. The source must be an official university/department page (e.g., faculty profile, department news/announcement)."
    )

    # Top-tier US university verification via rankings
    top_leaf = evaluator.add_leaf(
        id=f"fm_{idx+1}_top_tier_us_university",
        desc="Institution is a top-tier U.S. research university as defined by appearing in a major ranking (CSRankings/US News/QS), with a verifiable citation/link indicating inclusion.",
        parent=fm_node,
        critical=True
    )
    top_claim = (
        f"The institution {member.institution} appears in a major CS ranking list (CSRankings.org, US News Computer Science, or QS World Rankings)."
    )
    await evaluator.verify(
        claim=top_claim,
        node=top_leaf,
        sources=member.ranking_urls,  # multi-URL verification; fails if empty
        additional_instruction="Verify that the institution name is listed on the ranking page(s). Fuzzy match institution names reasonably (e.g., abbreviations, 'Univ.' vs 'University'). Any one of the provided ranking links suffices."
    )

    # Research area verification (ML/NLP/CV explicitly on profile)
    area_leaf = evaluator.add_leaf(
        id=f"fm_{idx+1}_research_area_in_scope",
        desc="Primary research area(s) explicitly include at least one of: machine learning, natural language processing, or computer vision (as stated on the official university profile).",
        parent=fm_node,
        critical=True
    )
    area_text = ", ".join(member.research_areas) if member.research_areas else ""
    area_claim = (
        f"The official faculty profile for {member.name} explicitly lists at least one of: machine learning, natural language processing, or computer vision. Current extracted areas: [{area_text}]."
    )
    await evaluator.verify(
        claim=area_claim,
        node=area_leaf,
        sources=member.faculty_profile_url,
        additional_instruction="Accept synonyms/variants like 'ML', 'NLP', 'CV', 'Deep Learning', 'Language Models', 'Visual Recognition' if clearly part of the AI subfields. The statement must be explicitly supported by the profile."
    )

    # Faculty profile URL confirms appointment
    profile_leaf = evaluator.add_leaf(
        id=f"fm_{idx+1}_faculty_profile_url",
        desc="Provide a direct URL to the official CS department faculty profile page that confirms the appointment.",
        parent=fm_node,
        critical=True
    )
    profile_claim = (
        f"This URL is an official CS department (or School of Computing) faculty profile page for {member.name} at {member.institution}, confirming the appointment."
    )
    await evaluator.verify(
        claim=profile_claim,
        node=profile_leaf,
        sources=member.faculty_profile_url,
        additional_instruction="Confirm that the page is clearly a university CS/Computing faculty profile, naming the person and institution, and indicating their faculty appointment."
    )

    # PhD-granting institution verification
    phd_leaf = evaluator.add_leaf(
        id=f"fm_{idx+1}_phd_granting_institution",
        desc="Provide PhD-granting institution as stated on an official profile/biography page.",
        parent=fm_node,
        critical=True
    )
    phd_sources = member.phd_evidence_urls if member.phd_evidence_urls else ([member.faculty_profile_url] if member.faculty_profile_url else [])
    phd_claim = (
        f"The official sources state that {member.name} earned a PhD from {member.phd_institution}."
    )
    await evaluator.verify(
        claim=phd_claim,
        node=phd_leaf,
        sources=phd_sources,
        additional_instruction="Look for 'PhD'/'Doctorate' in the profile/biography page and ensure the institution matches the provided value. Reasonable name variants are acceptable."
    )

    # Google Scholar profile verification
    gs_leaf = evaluator.add_leaf(
        id=f"fm_{idx+1}_google_scholar_profile_url",
        desc="Provide a direct URL to the faculty member's Google Scholar profile.",
        parent=fm_node,
        critical=True
    )
    gs_claim = (
        f"This URL is the Google Scholar profile page of {member.name}."
    )
    await evaluator.verify(
        claim=gs_claim,
        node=gs_leaf,
        sources=member.google_scholar_url,
        additional_instruction="Confirm that the page is a Google Scholar profile and the display name corresponds to the faculty member (allow reasonable variants)."
    )

    # Post-2020 publication verification
    pub_leaf = evaluator.add_leaf(
        id=f"fm_{idx+1}_post_2020_publication",
        desc="Provide at least one publication authored by the faculty member published after 2020, including title, venue, and a link on Google Scholar or DBLP.",
        parent=fm_node,
        critical=True
    )
    pub_title = member.publication.title if member.publication else ""
    pub_venue = member.publication.venue if member.publication else ""
    pub_year = member.publication.year if member.publication else ""
    pub_claim = (
        f"The provided source(s) show a publication titled '{pub_title}' authored by {member.name}, published after 2020 (year: {pub_year}) in venue '{pub_venue}'."
    )
    await evaluator.verify(
        claim=pub_claim,
        node=pub_leaf,
        sources=(member.publication.urls if member.publication and member.publication.urls else []),
        additional_instruction="Verify that the paper title appears on the page, it lists the faculty member as an author, and the publication year is 2021 or later. Accept Google Scholar or DBLP pages."
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
    Evaluate an answer for the new CS assistant professor (2024-2025) task.
    """
    # Initialize evaluator (root as parallel aggregation, non-critical to allow partial credit).
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

    # Extract structured faculty info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_faculty(),
        template_class=FacultyExtraction,
        extraction_name="faculty_extraction"
    )

    # Keep only the first four members for evaluation; pad with empty entries if fewer than four
    members = first_four_members(extraction.members)
    while len(members) < 4:
        members.append(FacultyMember())

    # Add ground truth expectations (for summary context, not used as "truth" in judging)
    evaluator.add_ground_truth({
        "required_count": 4,
        "time_window": "Fall 2024 or later (2024–2025 academic year)",
        "required_areas": ["machine learning", "natural language processing", "computer vision"],
        "ranking_sources": ["CSRankings.org", "US News Computer Science", "QS World Rankings"]
    }, gt_type="task_requirements")

    # Critical check: exactly four distinct faculty members provided (no duplicates).
    four_distinct_node = evaluator.add_custom_node(
        result=are_four_distinct(members),
        id="four_distinct_faculty_members_provided",
        desc="Provide exactly four distinct faculty members (no duplicates).",
        parent=root,
        critical=True
    )

    # Build verification subtrees for each faculty member
    for i, m in enumerate(members):
        await verify_faculty_member(evaluator, root, m, i)

    # Return structured summary
    return evaluator.get_summary()