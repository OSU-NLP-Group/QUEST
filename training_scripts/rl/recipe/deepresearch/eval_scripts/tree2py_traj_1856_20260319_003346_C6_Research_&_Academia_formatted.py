import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "r1_ai_ml_faculty_selection"
TASK_DESCRIPTION = """
Identify three tenure-track faculty members (Assistant Professor, Associate Professor, or Full Professor) at R1 universities in the United States who specialize in Artificial Intelligence or Machine Learning research. For each faculty member, the following criteria must be satisfied:

1. University Affiliation: The faculty member must be affiliated with an R1 university according to the Carnegie Classification (Doctoral Universities - Very High Research Activity).

2. Academic Position: The faculty member must hold a tenure-track position in a Computer Science department or closely related program.

3. Research Specialization: The faculty member's primary research focus must be in Artificial Intelligence, Machine Learning, or closely related subfields.

4. Publication Record:
   - The faculty member must have published at least 10 papers in peer-reviewed conferences or journals within the last 5 years (2021-2026).
   - At least 5 of these publications must be in top-tier venues (CORE A* or A ranked conferences, or equivalent high-impact journals).
   - The faculty member must have an h-index of at least 10 (verifiable through Google Scholar or similar databases).

5. Recent Research Activity: The faculty member must have published at least one paper within the most recent 2 years (2025-2026).

6. PhD Student Supervision: The faculty member must be currently supervising or have recently supervised at least one PhD student.

For each identified faculty member, provide:
- Full name and academic title
- University affiliation and department
- Link to their official university faculty profile
- Link to their Google Scholar profile or DBLP page
- Current h-index value
- Evidence of PhD student supervision (e.g., link to research group page or student listings)
- At least 3 examples of their publications from the last 5 years, including venue names and years
- At least 2 examples of their publications in top-tier venues (CORE A* or A), with verification of venue ranking

All information must be verifiable through publicly accessible web sources.
"""

CURRENT_YEAR = 2026
WINDOW_START_YEAR = 2021
RECENT_YEARS = {2025, 2026}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PublicationItem(BaseModel):
    title: Optional[str] = None
    venue: Optional[str] = None
    year: Optional[str] = None
    url: Optional[str] = None
    ranking: Optional[str] = None  # e.g., "CORE A*", "CORE A", "Top journal", etc.
    ranking_url: Optional[str] = None


class FacultyItem(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    university: Optional[str] = None
    department: Optional[str] = None
    official_profile_url: Optional[str] = None
    scholar_or_dblp_url: Optional[str] = None
    h_index: Optional[str] = None
    supervision_evidence_url: Optional[str] = None
    publications_recent: List[PublicationItem] = Field(default_factory=list)
    top_tier_examples: List[PublicationItem] = Field(default_factory=list)
    r1_verification_urls: List[str] = Field(default_factory=list)
    extra_sources: List[str] = Field(default_factory=list)


class FacultyListExtraction(BaseModel):
    faculty: List[FacultyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_faculty_list() -> str:
    return f"""
    From the answer, extract a list of faculty members described. For each faculty, extract the following fields exactly as stated:

    - name: Full name of the faculty member.
    - title: Academic title (e.g., Assistant Professor, Associate Professor, Professor). Avoid non–tenure-track titles such as Research/Teaching/Clinical/Adjunct.
    - university: University name.
    - department: Department or closely related program (e.g., Computer Science, Computer Engineering, ECE, Data Science, Robotics, Statistics if clearly computing-related).
    - official_profile_url: URL of the person's official university faculty profile page.
    - scholar_or_dblp_url: URL of their Google Scholar profile or DBLP page (one URL).
    - h_index: The stated current h-index number in the answer text (extract as string; do not compute).
    - supervision_evidence_url: URL that evidences PhD student supervision (e.g., research group page or student listing). If multiple are present, pick one most relevant.
    - publications_recent: An array of example publications from 2021–{CURRENT_YEAR} mentioned in the answer. For each, extract:
        * title
        * venue (conference/journal name)
        * year
        * url (if provided)
        * ranking (if the answer states a ranking like CORE A*/A or equivalent)
        * ranking_url (a URL that verifies the ranking, if provided in the answer)
    - top_tier_examples: An array of at least two example publications explicitly claimed as top-tier (CORE A*/A or equivalent), if provided. For each, include the same fields as publications_recent and especially ranking and ranking_url if given.
    - r1_verification_urls: Any URLs in the answer that verify the university is an R1 (Carnegie "Very High Research Activity") and in the United States (e.g., a Carnegie page or Wikipedia page stating R1).
    - extra_sources: Any additional URLs mentioned that could help verify claims (e.g., lab pages, PDFs, venue pages, press releases, ranking lists). Exclude duplicates of the previous URL fields.

    Rules:
    - Extract only what is explicitly present in the answer. Do not infer or invent.
    - Include URLs exactly as written (accept plain or markdown links).
    - If a field is missing, set it to null (or empty list for arrays).
    - Preserve multiple faculty as a 'faculty' array; do not merge across people.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_url_list(*url_lists: List[Optional[str]] | List[List[Optional[str]]]) -> List[str]:
    urls: List[str] = []
    for ul in url_lists:
        if isinstance(ul, list):
            for u in ul:
                if isinstance(u, list):
                    for uu in u:
                        if uu and isinstance(uu, str) and uu.strip():
                            urls.append(uu.strip())
                else:
                    if u and isinstance(u, str) and u.strip():
                        urls.append(u.strip())
        elif isinstance(ul, str):
            if ul.strip():
                urls.append(ul.strip())
    # Deduplicate preserving order
    seen = set()
    uniq: List[str] = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def _parse_int_year(y: Optional[str]) -> Optional[int]:
    if not y:
        return None
    try:
        # Extract first 4-digit number
        for tok in y.replace("/", " ").replace("-", " ").split():
            if len(tok) == 4 and tok.isdigit():
                val = int(tok)
                if 1900 <= val <= 2100:
                    return val
        # Fallback: full string int
        v = int(y)
        if 1900 <= v <= 2100:
            return v
    except Exception:
        return None
    return None


def _is_in_recent_window(pub: PublicationItem) -> bool:
    yr = _parse_int_year(pub.year)
    return yr is not None and WINDOW_START_YEAR <= yr <= CURRENT_YEAR


def _is_in_most_recent_two_years(pub: PublicationItem) -> bool:
    yr = _parse_int_year(pub.year)
    return yr in RECENT_YEARS if yr is not None else False


def _unique_by_name(faculty: List[FacultyItem]) -> List[FacultyItem]:
    seen = set()
    out: List[FacultyItem] = []
    for f in faculty:
        key = (f.name or "").strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


# --------------------------------------------------------------------------- #
# Verification for one faculty                                                #
# --------------------------------------------------------------------------- #
async def verify_one_faculty(
    evaluator: Evaluator,
    parent_node,
    faculty: FacultyItem,
    index: int
) -> None:
    """
    Build evaluation subtree for a single faculty member and run all verifications.
    All nodes here are marked critical because the rubric requires every constraint to be satisfied.
    """
    person_name = faculty.name or f"Faculty #{index+1}"
    uni = faculty.university or "the stated university"
    dept = faculty.department or "the stated department"
    title = faculty.title or "the stated title"

    # Parent node for this faculty (critical; parallel aggregation within)
    fac_node = evaluator.add_parallel(
        id=f"Faculty_Member_{index+1}",
        desc=f"Evaluation of the {index+1}th evaluated faculty member against all constraints and required outputs.",
        parent=parent_node,
        critical=True
    )

    # ------------------ Eligibility Constraints (critical parallel) ------------------ #
    elig_node = evaluator.add_parallel(
        id=f"F{index+1}_Eligibility_Constraints",
        desc="Faculty member meets all eligibility constraints (affiliation, rank, research area, publications, activity, supervision).",
        parent=fac_node,
        critical=True
    )

    # 1) R1 US Affiliation
    r1_leaf = evaluator.add_leaf(
        id=f"F{index+1}_R1_US_Affiliation_Verified",
        desc="Affiliated with a US-based Carnegie R1 university (Doctoral Universities: Very High Research Activity), verifiable via public sources.",
        parent=elig_node,
        critical=True
    )
    r1_sources = _normalize_url_list(
        faculty.official_profile_url,
        faculty.r1_verification_urls,
        faculty.extra_sources
    )
    r1_claim = (
        f"{person_name} is affiliated with {uni}, which is in the United States and is classified as "
        f"R1 (Doctoral Universities: Very High Research Activity) per Carnegie Classification."
    )
    await evaluator.verify(
        claim=r1_claim,
        node=r1_leaf,
        sources=r1_sources,
        additional_instruction="Look for explicit evidence that the university is in the US and is classified as R1 (Very High Research Activity) by Carnegie. If none of the provided URLs explicitly support R1 status, mark as not supported."
    )

    # 2) Tenure-track rank in CS or related program
    tenure_leaf = evaluator.add_leaf(
        id=f"F{index+1}_Tenure_Track_CS_or_Related_Verified",
        desc="Holds a tenure-track rank (Assistant/Associate/Full Professor) in a CS department or closely related program, via official sources.",
        parent=elig_node,
        critical=True
    )
    tenure_claim = (
        f"On the official profile, {person_name} holds a tenure-track position "
        f"({title}) in {dept}, which is a Computer Science department or a closely related computing program."
    )
    await evaluator.verify(
        claim=tenure_claim,
        node=tenure_leaf,
        sources=faculty.official_profile_url,
        additional_instruction="Accept Assistant/Associate/Professor as tenure-track unless the page explicitly indicates non-tenure (e.g., Research/Teaching/Clinical/Adjunct). Accept closely related programs (e.g., ECE, Data Science, Robotics, Computer Engineering, Statistics if computing-focused)."
    )

    # 3) AI/ML specialization
    aiml_leaf = evaluator.add_leaf(
        id=f"F{index+1}_AI_ML_Specialization_Verified",
        desc="Primary research focus is AI/ML (or closely related subfields), verifiable via profile/publications.",
        parent=elig_node,
        critical=True
    )
    aiml_sources = _normalize_url_list(
        faculty.official_profile_url, faculty.scholar_or_dblp_url
    )
    aiml_claim = (
        f"{person_name}'s primary research focus is in Artificial Intelligence or Machine Learning, "
        f"including closely related subfields (e.g., deep learning, NLP, computer vision, reinforcement learning)."
    )
    await evaluator.verify(
        claim=aiml_claim,
        node=aiml_leaf,
        sources=aiml_sources,
        additional_instruction="Use the official profile and/or Scholar/DBLP page. Look for explicit mentions of AI/ML or obviously related subfields."
    )

    # 4) >=10 publications in 2021–2026
    pubs10_leaf = evaluator.add_leaf(
        id=f"F{index+1}_Publications_10_in_2021_2026_Verified",
        desc="Has at least 10 peer-reviewed papers during 2021–2026, verifiable via public sources.",
        parent=elig_node,
        critical=True
    )
    pubs10_sources = _normalize_url_list(faculty.scholar_or_dblp_url, faculty.extra_sources)
    pubs10_claim = f"From 2021 through {CURRENT_YEAR}, {person_name} has at least 10 peer-reviewed conference or journal publications."
    await evaluator.verify(
        claim=pubs10_claim,
        node=pubs10_leaf,
        sources=pubs10_sources,
        additional_instruction="Count publications between 2021 and 2026 inclusive. Prefer peer-reviewed conferences/journals; ignore non-refereed items if clearly labeled."
    )

    # 5) >=5 top-tier (CORE A*/A or equivalent) during 2021–2026
    top5_leaf = evaluator.add_leaf(
        id=f"F{index+1}_Top_Tier_Publications_5_Verified",
        desc="At least 5 of the 2021–2026 publications are in top-tier venues (CORE A*/A, or equivalent high-impact journals).",
        parent=elig_node,
        critical=True
    )
    # Prefer ranking URLs and venue pages if provided
    ranking_urls = [p.ranking_url for p in (faculty.top_tier_examples or []) if p.ranking_url]  # may be limited
    top5_sources = _normalize_url_list(faculty.scholar_or_dblp_url, ranking_urls, faculty.extra_sources)
    top5_claim = (
        f"At least five of {person_name}'s publications in 2021–{CURRENT_YEAR} are in top-tier venues "
        f"(CORE A* or A) or equivalent high-impact journals."
    )
    await evaluator.verify(
        claim=top5_claim,
        node=top5_leaf,
        sources=top5_sources,
        additional_instruction="Use provided venue ranking sources (e.g., CORE portal) or explicit statements on authoritative pages. If ranking support is not found in provided URLs, treat as not supported."
    )

    # 6) h-index >= 10
    hidx_leaf = evaluator.add_leaf(
        id=f"F{index+1}_H_Index_At_Least_10_Verified",
        desc="Has h-index ≥ 10, verifiable via Google Scholar or similar database.",
        parent=elig_node,
        critical=True
    )
    hidx_claim = f"{person_name} has an h-index of at least 10."
    await evaluator.verify(
        claim=hidx_claim,
        node=hidx_leaf,
        sources=faculty.scholar_or_dblp_url,
        additional_instruction="Check Google Scholar (preferred) or similar databases. If page shows h-index < 10 or no h-index is visible, mark as not supported."
    )

    # 7) At least one publication in 2025–2026
    recent_leaf = evaluator.add_leaf(
        id=f"F{index+1}_Recent_Paper_2025_2026_Verified",
        desc="Has at least one publication in 2025–2026, verifiable via public sources.",
        parent=elig_node,
        critical=True
    )
    recent_claim = f"{person_name} has at least one publication in 2025 or 2026."
    await evaluator.verify(
        claim=recent_claim,
        node=recent_leaf,
        sources=faculty.scholar_or_dblp_url,
        additional_instruction="Use the publications list on Scholar/DBLP to confirm at least one 2025 or 2026 item."
    )

    # 8) PhD supervision
    phd_leaf = evaluator.add_leaf(
        id=f"F{index+1}_PhD_Supervision_Verified",
        desc="Currently supervising or recently supervised at least one PhD student, with public evidence.",
        parent=elig_node,
        critical=True
    )
    phd_sources = _normalize_url_list(faculty.supervision_evidence_url, faculty.official_profile_url, faculty.extra_sources)
    phd_claim = f"{person_name} currently supervises or has recently supervised at least one PhD student."
    await evaluator.verify(
        claim=phd_claim,
        node=phd_leaf,
        sources=phd_sources,
        additional_instruction="Look for explicit student listings, lab pages, or supervision statements. If not explicitly present on provided URLs, mark as not supported."
    )

    # ------------------ Required Output Fields (critical parallel) ------------------ #
    out_node = evaluator.add_parallel(
        id=f"F{index+1}_Required_Output_Fields",
        desc="All required output fields/links/examples are provided for this faculty member.",
        parent=fac_node,
        critical=True
    )

    # Name and Title provided
    evaluator.add_custom_node(
        result=bool((faculty.name or "").strip()) and bool((faculty.title or "").strip()),
        id=f"F{index+1}_Name_and_Title_Provided",
        desc="Full name and academic title are provided.",
        parent=out_node,
        critical=True
    )

    # Affiliation and Department provided
    evaluator.add_custom_node(
        result=bool((faculty.university or "").strip()) and bool((faculty.department or "").strip()),
        id=f"F{index+1}_Affiliation_and_Department_Provided",
        desc="University affiliation and department/program are provided.",
        parent=out_node,
        critical=True
    )

    # Official profile link provided
    evaluator.add_custom_node(
        result=bool((faculty.official_profile_url or "").strip()),
        id=f"F{index+1}_Official_Profile_Link_Provided",
        desc="Link to official university faculty profile is provided.",
        parent=out_node,
        critical=True
    )

    # Scholar or DBLP link provided
    evaluator.add_custom_node(
        result=bool((faculty.scholar_or_dblp_url or "").strip()),
        id=f"F{index+1}_Scholar_or_DBLP_Link_Provided",
        desc="Link to Google Scholar profile or DBLP page is provided.",
        parent=out_node,
        critical=True
    )

    # h-index value stated (as a value in the answer)
    evaluator.add_custom_node(
        result=bool((faculty.h_index or "").strip()),
        id=f"F{index+1}_H_Index_Value_Stated",
        desc="Current h-index value is explicitly stated.",
        parent=out_node,
        critical=True
    )

    # PhD supervision evidence link provided
    evaluator.add_custom_node(
        result=bool((faculty.supervision_evidence_url or "").strip()),
        id=f"F{index+1}_PhD_Supervision_Evidence_Link_Provided",
        desc="A public link evidencing PhD student supervision is provided.",
        parent=out_node,
        critical=True
    )

    # Three recent publication examples provided (2021–CURRENT_YEAR) with venue and year
    pubs_in_window = [p for p in faculty.publications_recent if _is_in_recent_window(p)]
    pubs_with_venue_year = [p for p in pubs_in_window if (p.venue and p.venue.strip()) and (p.year and p.year.strip())]
    evaluator.add_custom_node(
        result=len(pubs_with_venue_year) >= 3,
        id=f"F{index+1}_Three_Recent_Publications_Examples_Provided",
        desc=f"At least 3 example publications from 2021–{CURRENT_YEAR} are provided, each with venue name and year.",
        parent=out_node,
        critical=True
    )

    # Two top-tier publication examples provided WITH venue-ranking verification
    top_examples = faculty.top_tier_examples or []
    # Prepare verification leaf that uses provided ranking URLs (if any)
    top2_leaf = evaluator.add_leaf(
        id=f"F{index+1}_Two_Top_Tier_Publications_Examples_Provided",
        desc="At least 2 example top-tier (CORE A*/A or equivalent) publications are provided, with venue-ranking verification.",
        parent=out_node,
        critical=True
    )
    # Build claim string listing up to two examples
    example_strs: List[str] = []
    ranking_support_urls: List[str] = []
    for p in top_examples[:2]:
        ttl = (p.title or "").strip()
        ven = (p.venue or "").strip()
        yr = (p.year or "").strip()
        if ttl or ven:
            example_strs.append(f"'{ttl}' at {ven} ({yr})")
        if p.ranking_url:
            ranking_support_urls.append(p.ranking_url)
        if p.url:
            ranking_support_urls.append(p.url)
    # Combine with any extra ranking sources the answer may provide
    top2_sources = _normalize_url_list(ranking_support_urls, faculty.extra_sources)

    top2_claim = (
        "The following example publications are in top-tier venues (CORE A*/A or equivalent high-impact journals): "
        + ("; ".join(example_strs) if example_strs else "No examples provided.")
    )
    await evaluator.verify(
        claim=top2_claim,
        node=top2_leaf,
        sources=top2_sources,
        additional_instruction="Use the provided ranking URLs or authoritative sources to confirm that the listed venues are CORE A* or A (or equivalent high-impact journals). If ranking evidence cannot be found on the provided URLs, mark as not supported."
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
    Evaluate an answer for the AI/ML R1 faculty identification task.
    """
    # Initialize evaluator with a sequential root to gate downstream evaluation on prerequisites
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

    # Record ground-truth constraints for transparency
    evaluator.add_ground_truth({
        "constraints": {
            "R1_US_only": True,
            "tenure_track_required": True,
            "research_area": "AI/ML or closely related",
            "pub_window_years": [WINDOW_START_YEAR, CURRENT_YEAR],
            "pub_min_count": 10,
            "top_tier_min_count": 5,
            "recent_years": list(sorted(RECENT_YEARS)),
            "recent_min_count": 1,
            "h_index_min": 10,
            "supervision_required": True,
            "num_faculty_required": 3
        }
    }, gt_type="rubric_constraints")

    # Extract structured information from the agent's answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_faculty_list(),
        template_class=FacultyListExtraction,
        extraction_name="faculty_extraction"
    )

    # Select first three distinct faculty by name
    uniq_faculty = _unique_by_name(extracted.faculty or [])
    selected = uniq_faculty[:3]
    evaluator.add_custom_info(
        info={
            "extracted_count": len(extracted.faculty or []),
            "distinct_count": len(uniq_faculty),
            "selected_names": [f.name for f in selected]
        },
        info_type="selection_info"
    )

    # 1) At least three distinct faculty provided (critical)
    atleast3_node = evaluator.add_custom_node(
        result=len(uniq_faculty) >= 3,
        id="At_Least_Three_Distinct_Faculty_Provided",
        desc="At least three distinct faculty members are provided (no duplicates). Evaluation applies to the first three listed.",
        parent=root,
        critical=True
    )

    # 2) Evaluate each of the three selected faculty members (critical, parallel aggregation)
    eval_all_node = evaluator.add_parallel(
        id="Evaluate_Three_Faculty",
        desc="Evaluate each of the three selected faculty members independently against constraints and required outputs.",
        parent=root,
        critical=True
    )

    # Build and run verification subtrees for up to 3 faculty
    for idx in range(3):
        fac = selected[idx] if idx < len(selected) else FacultyItem()  # Safety, though node-1 gates this
        await verify_one_faculty(evaluator, eval_all_node, fac, idx)

    # Return evaluator summary
    return evaluator.get_summary()