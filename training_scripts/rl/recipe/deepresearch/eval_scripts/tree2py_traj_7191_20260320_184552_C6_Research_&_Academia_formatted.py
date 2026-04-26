import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "top_astro_universities_eclipse_2026"
TASK_DESCRIPTION = (
    "Identify four research universities in the United States that satisfy ALL of the following criteria: "
    "(1) Have an astronomy or astrophysics department or program ranked in the top 10 nationally according to verifiable 2026 rankings, "
    "(2) Are located in regions where the March 3, 2026 total lunar eclipse will be fully visible during the entire totality phase "
    "(from 11:04:26 UTC to 12:02:49 UTC), "
    "(3) Operate or are affiliated with at least one astronomical observatory. "
    "For each of the four universities, provide the following information: the local time range (start and end times) when the totality phase will be observable "
    "from the university's location, the name of at least one astronomical observatory operated by or affiliated with the university, the official website URL of "
    "the university's astronomy or astrophysics department, and evidence of at least one astronomical research publication, paper, or research project from the "
    "university's astronomy department dated 2025 or 2026. Include reference URLs for all factual claims."
)

RESEARCH_TASK_COMPLETION_DESC = (
    "Four U.S. universities with top-ranked astronomy programs and full March 3, 2026 eclipse visibility are identified, "
    "each with complete required information"
)

ECLIPSE_DATE_UTC = "March 3, 2026"
ECLIPSE_UTC_START = "11:04:26 UTC"
ECLIPSE_UTC_END = "12:02:49 UTC"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PublicationInfo(BaseModel):
    title: Optional[str] = None
    date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class UniversityEntry(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    ranking_statement: Optional[str] = None
    ranking_source_urls: List[str] = Field(default_factory=list)

    eclipse_visibility_source_urls: List[str] = Field(default_factory=list)
    eclipse_timing_source_urls: List[str] = Field(default_factory=list)
    eclipse_totality_local_start: Optional[str] = None
    eclipse_totality_local_end: Optional[str] = None

    observatory_names: List[str] = Field(default_factory=list)
    observatory_source_urls: List[str] = Field(default_factory=list)

    department_url: Optional[str] = None

    research_publications: List[PublicationInfo] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    From the answer, extract details for up to FOUR (4) U.S. research universities that claim to meet ALL these criteria:
    - The astronomy or astrophysics department/program is ranked in the TOP 10 nationally according to verifiable 2026 rankings.
    - The March 3, 2026 total lunar eclipse totality phase is FULLY visible from their location for the entire totality window (11:04:26 UTC to 12:02:49 UTC).
    - The university operates or is affiliated with at least one astronomical observatory.

    For EACH university (limit to the first four mentioned), extract the following fields:
    - name: University name (string).
    - location: City/State or geographic descriptor given in the answer (string, optional).
    - ranking_statement: Any brief text the answer gives about the 2026 ranking (string, optional).
    - ranking_source_urls: All URLs the answer cites for the 2026 top-10 ranking claim (array of URLs).
    - eclipse_visibility_source_urls: All URLs the answer cites for eclipse visibility/where the eclipse is visible (array of URLs).
    - eclipse_timing_source_urls: All URLs the answer cites that explicitly provide local times for totality at the university's location (array of URLs).
    - eclipse_totality_local_start: The local time the answer claims totality STARTS at the university's location (string as presented).
    - eclipse_totality_local_end: The local time the answer claims totality ENDS at the university's location (string as presented).
    - observatory_names: Names of at least one observatory the university operates or is affiliated with (array of strings).
    - observatory_source_urls: All URLs the answer cites to support the observatory operation or affiliation (array of URLs).
    - department_url: The official Astronomy/Astrophysics department website URL for the university (single URL).
    - research_publications: An array of at least one research item (publication/paper/project) the answer cites for 2025 or 2026 from the university's astronomy unit. Each item should include:
        • title (string, if provided),
        • date (string as presented, if provided),
        • urls (array of URLs that serve as evidence for that item).

    IMPORTANT:
    - Only extract information explicitly present in the answer. Do not invent.
    - For all URL fields, extract only valid URLs explicitly mentioned in the answer (plain links or markdown).
    - If an item is missing in the answer, set null (for strings) or empty array (for arrays).
    - Return up to 4 universities in the 'universities' array, in the same order as in the answer.
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"][n] if n < 6 else f"{n+1}th"


def pick_first_nonempty(items: List[str]) -> Optional[str]:
    for x in items:
        if x and isinstance(x, str) and x.strip():
            return x.strip()
    return None


def has_year_2025_or_2026(s: Optional[str]) -> bool:
    if not s:
        return False
    s_lower = s.lower()
    return "2025" in s_lower or "2026" in s_lower


def pick_recent_publication(pubs: List[PublicationInfo]) -> Optional[PublicationInfo]:
    for p in pubs:
        if has_year_2025_or_2026(p.date) or has_year_2025_or_2026(p.title):
            return p
    return pubs[0] if pubs else None


# --------------------------------------------------------------------------- #
# Verification subroutine for a single university                             #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityEntry,
    index: int,
) -> None:
    ui = index + 1
    uni_name = uni.name or "the university"

    # University container (non-critical; the sub-groups are critical)
    univ_node = evaluator.add_parallel(
        id=f"University_{ui}",
        desc=f"{ordinal(index)} qualifying university with all required details",
        parent=parent_node,
        critical=False,
    )

    # 1) Qualification (all critical checks inside)
    qual_node = evaluator.add_parallel(
        id=f"University_{ui}_Qualification",
        desc="University meets all qualification criteria",
        parent=univ_node,
        critical=True,
    )

    # 1.a) Ranking reference provided (existence check)
    evaluator.add_custom_node(
        result=bool(uni.ranking_source_urls),
        id=f"University_{ui}_Ranking_Reference",
        desc="URL reference provided for the ranking information",
        parent=qual_node,
        critical=True,
    )

    # 1.b) Ranking top-10 (2026, U.S., astronomy/astrophysics)
    ranking_leaf = evaluator.add_leaf(
        id=f"University_{ui}_Ranking",
        desc="University has a top-10 ranked astronomy or astrophysics program according to verifiable 2026 rankings",
        parent=qual_node,
        critical=True,
    )
    ranking_claim = (
        f"According to the provided 2026 ranking source(s), the astronomy or astrophysics program at {uni_name} "
        f"is ranked within the top 10 in the United States for 2026."
    )
    await evaluator.verify(
        claim=ranking_claim,
        node=ranking_leaf,
        sources=uni.ranking_source_urls,
        additional_instruction=(
            "Evaluate only if the page(s) provide 2026 rankings. Accept 'Astronomy', 'Astrophysics', or 'Astronomy & Astrophysics'. "
            "It must be a U.S. national ranking; if the source is global, determine whether the U.S.-only position is top 10; "
            "if this cannot be determined, mark as not supported. Ranks 1-10 inclusive count as 'top-10'."
        ),
    )

    # 1.c) Eclipse visibility reference provided
    evaluator.add_custom_node(
        result=bool(uni.eclipse_visibility_source_urls),
        id=f"University_{ui}_Eclipse_Visibility_Reference",
        desc="URL reference provided for eclipse visibility information",
        parent=qual_node,
        critical=True,
    )

    # 1.d) Eclipse full visibility across totality (UTC window)
    eclipse_vis_leaf = evaluator.add_leaf(
        id=f"University_{ui}_Eclipse_Visibility",
        desc=(
            "University location has full visibility of the March 3, 2026 total lunar eclipse totality phase "
            "(can observe from totality start 11:04:26 UTC through totality end 12:02:49 UTC)"
        ),
        parent=qual_node,
        critical=True,
    )
    eclipse_vis_claim = (
        f"From {uni_name}'s location, the totality of the {ECLIPSE_DATE_UTC} total lunar eclipse is visible for the entire window "
        f"from {ECLIPSE_UTC_START} to {ECLIPSE_UTC_END} (UTC), i.e., the Moon is above the horizon during the full totality interval."
    )
    await evaluator.verify(
        claim=eclipse_vis_claim,
        node=eclipse_vis_leaf,
        sources=uni.eclipse_visibility_source_urls,
        additional_instruction=(
            "Use the page's visibility data for the university's city/region. The result should indicate totality is visible "
            "for the whole totality interval, not just partially. If the page shows that either the start or end of totality is "
            "not visible (e.g., moonset/sunrise interrupts it), mark as not supported."
        ),
    )

    # 1.e) Observatory affiliation reference provided
    evaluator.add_custom_node(
        result=bool(uni.observatory_source_urls),
        id=f"University_{ui}_Observatory_Affiliation_Reference",
        desc="URL reference provided for observatory affiliation information",
        parent=qual_node,
        critical=True,
    )

    # 1.f) Observatory affiliation truth
    observatory_leaf = evaluator.add_leaf(
        id=f"University_{ui}_Observatory",
        desc="University operates or is affiliated with at least one astronomical observatory",
        parent=qual_node,
        critical=True,
    )
    obs_name = pick_first_nonempty(uni.observatory_names)
    if obs_name:
        observatory_claim = (
            f"{uni_name} operates or is formally affiliated with the astronomical observatory '{obs_name}'."
        )
    else:
        observatory_claim = (
            f"{uni_name} operates or is formally affiliated with at least one astronomical observatory."
        )
    await evaluator.verify(
        claim=observatory_claim,
        node=observatory_leaf,
        sources=uni.observatory_source_urls,
        additional_instruction=(
            "The page(s) should clearly indicate operation or official affiliation (e.g., operates, manages, partner institution, "
            "member institution). Generic mentions of visits or unrelated outreach are insufficient."
        ),
    )

    # 2) Eclipse timing (local start/end) — critical
    timing_node = evaluator.add_parallel(
        id=f"University_{ui}_Eclipse_Timing",
        desc="Local time range for totality viewing from the university's location",
        parent=univ_node,
        critical=True,
    )

    timing_sources = uni.eclipse_timing_source_urls if uni.eclipse_timing_source_urls else uni.eclipse_visibility_source_urls
    has_timing_ref = bool(timing_sources)

    evaluator.add_custom_node(
        result=has_timing_ref,
        id=f"University_{ui}_Timing_Reference",
        desc="URL reference provided for eclipse timing information",
        parent=timing_node,
        critical=True,
    )

    # 2.a) Totality start local time
    start_leaf = evaluator.add_leaf(
        id=f"University_{ui}_Totality_Start",
        desc="Correct local time when totality begins at the university's location (converted from 11:04:26 UTC)",
        parent=timing_node,
        critical=True,
    )
    local_start = uni.eclipse_totality_local_start or ""
    start_claim = (
        f"For {uni_name}'s location, the totality of the {ECLIPSE_DATE_UTC} lunar eclipse begins at approximately "
        f"{local_start} local time (allow minor rounding or notation differences)."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=timing_sources if has_timing_ref else None,
        additional_instruction=(
            "Confirm that the referenced page lists the start of totality around the provided local time for the relevant city/region. "
            "Allow ±1 minute or differences in AM/PM/timezone abbreviations."
        ),
    )

    # 2.b) Totality end local time
    end_leaf = evaluator.add_leaf(
        id=f"University_{ui}_Totality_End",
        desc="Correct local time when totality ends at the university's location (converted from 12:02:49 UTC)",
        parent=timing_node,
        critical=True,
    )
    local_end = uni.eclipse_totality_local_end or ""
    end_claim = (
        f"For {uni_name}'s location, the totality of the {ECLIPSE_DATE_UTC} lunar eclipse ends at approximately "
        f"{local_end} local time (allow minor rounding or notation differences)."
    )
    await evaluator.verify(
        claim=end_claim,
        node=end_leaf,
        sources=timing_sources if has_timing_ref else None,
        additional_instruction=(
            "Confirm that the referenced page lists the end of totality around the provided local time for the relevant city/region. "
            "Allow ±1 minute or differences in AM/PM/timezone abbreviations."
        ),
    )

    # 3) Observatory details — critical
    obs_detail_node = evaluator.add_parallel(
        id=f"University_{ui}_Observatory_Details",
        desc="Name of at least one observatory operated by or affiliated with the university",
        parent=univ_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(uni.observatory_names),
        id=f"University_{ui}_Observatory_Name",
        desc="Specific name of the observatory is provided",
        parent=obs_detail_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(uni.observatory_source_urls),
        id=f"University_{ui}_Observatory_Reference",
        desc="URL reference provided for observatory information",
        parent=obs_detail_node,
        critical=True,
    )

    # 4) Department info — critical
    dept_node = evaluator.add_parallel(
        id=f"University_{ui}_Department_Info",
        desc="Official astronomy/astrophysics department website URL",
        parent=univ_node,
        critical=True,
    )

    if uni.department_url and uni.department_url.strip():
        dept_leaf = evaluator.add_leaf(
            id=f"University_{ui}_Website_URL",
            desc="Valid and accessible official department website URL is provided that leads to the correct department page for the specified university",
            parent=dept_node,
            critical=True,
        )
        dept_claim = (
            f"This page is the official website for the Astronomy/Astrophysics department (or Astronomy unit within Physics & Astronomy) of {uni_name}."
        )
        await evaluator.verify(
            claim=dept_claim,
            node=dept_leaf,
            sources=uni.department_url,
            additional_instruction=(
                "Verify that this is the official departmental page (typically on the university's domain) for Astronomy/Astrophysics. "
                "Accept names like 'Department of Astronomy', 'Department of Physics and Astronomy', or 'Astronomy & Astrophysics' as appropriate."
            ),
        )
    else:
        # If URL missing, fail this critical requirement explicitly
        evaluator.add_custom_node(
            result=False,
            id=f"University_{ui}_Website_URL",
            desc="Valid and accessible official department website URL is provided that leads to the correct department page for the specified university",
            parent=dept_node,
            critical=True,
        )

    # 5) Research evidence — critical
    research_node = evaluator.add_parallel(
        id=f"University_{ui}_Research_Evidence",
        desc="Evidence of astronomical research published by the university in 2025 or 2026",
        parent=univ_node,
        critical=True,
    )

    chosen_pub = pick_recent_publication(uni.research_publications)
    pub_urls: List[str] = chosen_pub.urls if (chosen_pub and chosen_pub.urls) else []

    # Research reference existence
    evaluator.add_custom_node(
        result=bool(pub_urls),
        id=f"University_{ui}_Research_Reference",
        desc="URL reference provided for the research evidence",
        parent=research_node,
        critical=True,
    )

    # Research publication verification
    pub_leaf = evaluator.add_leaf(
        id=f"University_{ui}_Publication",
        desc="At least one research publication, paper title, or research project from the astronomy department dated 2025-2026 is provided",
        parent=research_node,
        critical=True,
    )
    if chosen_pub and chosen_pub.title:
        pub_claim = (
            f"The page documents a research publication or project titled '{chosen_pub.title}' from the Astronomy/Astrophysics unit of "
            f"{uni_name}, with a publication or event date in 2025 or 2026."
        )
    else:
        pub_claim = (
            f"The page documents a research publication or project from the Astronomy/Astrophysics unit of {uni_name}, "
            f"with a publication or event date in 2025 or 2026."
        )
    await evaluator.verify(
        claim=pub_claim,
        node=pub_leaf,
        sources=pub_urls if pub_urls else None,
        additional_instruction=(
            "Confirm that the item is astronomy-related and clearly affiliated with the university (department page, author affiliation, "
            "or official news). The date on the page should show 2025 or 2026 (accept month/day variations). Accept arXiv/journal pages "
            "if affiliation or context indicates the university's astronomy unit involvement."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
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

    # Add top-level rubric node
    top_node = evaluator.add_parallel(
        id="Research_Task_Completion",
        desc=RESEARCH_TASK_COMPLETION_DESC,
        parent=root,
        critical=False,
    )

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Normalize list to exactly 4 entries (pad with empty if needed)
    universities: List[UniversityEntry] = list(extracted.universities[:4])
    while len(universities) < 4:
        universities.append(UniversityEntry())

    # Optional: record constraints as GT info
    evaluator.add_ground_truth({
        "eclipse_utc_window": {"start": ECLIPSE_UTC_START, "end": ECLIPSE_UTC_END, "date": ECLIPSE_DATE_UTC},
        "ranking_year_required": 2026,
        "top_n_required": 10,
        "num_universities_required": 4,
    })

    # Build verification tree for each university
    for idx in range(4):
        await verify_university(evaluator, top_node, universities[idx], idx)

    # Return structured evaluation summary
    return evaluator.get_summary()