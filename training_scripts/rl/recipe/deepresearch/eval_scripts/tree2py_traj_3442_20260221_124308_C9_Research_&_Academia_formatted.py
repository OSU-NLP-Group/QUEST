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
TASK_ID = "space_weather_consortium_universities"
TASK_DESCRIPTION = """
Identify 4 distinct U.S. universities that meet ALL of the following criteria for participating in a proposed multi-institutional space weather research consortium:

1. The university must be located in the United States.
2. The university must offer at least one doctoral (PhD) program in atmospheric sciences, space physics, astrophysics, geophysics, or a directly related interdisciplinary field.
3. The university must host or be affiliated with at least one research center, institute, or laboratory that conducts research in space weather, ionospheric physics, upper atmospheric studies, magnetospheric physics, or stratospheric dynamics.
4. The university must be eligible to receive research grants from the U.S. National Science Foundation (NSF).
5. The university must have documented research activities in at least one of the following areas: space weather prediction, geomagnetic storm studies, ionospheric research, polar vortex dynamics, or stratospheric warming.
6. The university's researchers must have published at least one peer-reviewed article in atmospheric science, space physics, or related fields between 2023 and 2026.
7. The university must have established policies or infrastructure supporting multi-institutional research collaborations.
8. The research center/institute mentioned in criterion 3 must have a publicly accessible website describing its research activities.
9. The doctoral program mentioned in criterion 2 must be currently accepting applications or have accepted students within the past 2 years (2024-2026).
10. The university must have documented faculty members actively conducting research in the relevant fields.

For each of the 4 universities, provide:
- The university name
- The name of the relevant doctoral program
- The name of the relevant research center/institute
- One example of a published research article (with title and year) from 2023-2026
- The URL of the research center's website
- The URL providing information about the doctoral program
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityEntry(BaseModel):
    university_name: Optional[str] = None
    university_homepage_url: Optional[str] = None

    phd_program_name: Optional[str] = None
    phd_program_url: Optional[str] = None

    research_center_name: Optional[str] = None
    research_center_url: Optional[str] = None

    publication_title: Optional[str] = None
    publication_year: Optional[str] = None
    publication_url: Optional[str] = None

    nsf_eligibility_url: Optional[str] = None
    collaboration_url: Optional[str] = None
    faculty_url: Optional[str] = None

    extra_support_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to FOUR distinct universities and their supporting details from the answer.
    For each university, return an object with the following fields:
    - university_name: The university’s full official name.
    - university_homepage_url: A URL to the university’s main website (if provided).
    - phd_program_name: Name of a doctoral (PhD) program in atmospheric sciences, space physics, astrophysics, geophysics, or a directly related interdisciplinary field.
    - phd_program_url: A URL to the PhD program page (this should be a specific program page if provided in the answer).
    - research_center_name: Name of a research center/institute/laboratory conducting space weather / ionospheric / upper atmosphere / magnetospheric / stratospheric dynamics research.
    - research_center_url: A URL to the research center’s website that describes its research.
    - publication_title: Title of one peer-reviewed publication (2023–2026) by the university’s researchers in atmospheric science, space physics, or related fields.
    - publication_year: The publication year (as presented in the answer), preferably 2023–2026.
    - publication_url: A URL to the publication page (journal/publisher/DOI/official repository if provided).
    - nsf_eligibility_url: A URL evidencing NSF eligibility (e.g., NSF award page, or a page indicating NSF-funded projects).
    - collaboration_url: A URL evidencing policies/infrastructure for multi-institution collaborations (e.g., research collaboration policy, consortia participation, sponsored research/partnership pages).
    - faculty_url: A URL evidencing faculty actively conducting research in the relevant fields (e.g., department/center people page).
    - extra_support_urls: Any additional URLs provided in the answer that support any of the above criteria (use only URLs that explicitly appear in the answer).

    Notes:
    - Extract only what is explicitly present in the answer. Do not invent URLs.
    - If a field is not present in the answer for a given university, return null for that field (or empty list for extra_support_urls).
    - Preserve the order given in the answer; if more than 4 universities are listed, only extract the first 4.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def dedup_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def gather_urls_for_uni(u: UniversityEntry) -> List[str]:
    return dedup_urls(
        [
            u.university_homepage_url,
            u.phd_program_url,
            u.research_center_url,
            u.publication_url,
            u.nsf_eligibility_url,
            u.collaboration_url,
            u.faculty_url,
            *(u.extra_support_urls or []),
        ]
    )


# --------------------------------------------------------------------------- #
# Verification logic per university                                           #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    root_parent,
    uni: UniversityEntry,
    index: int,
    prior_university_names: List[str],
) -> None:
    """
    Build the verification subtree for a single university according to the rubric.
    """

    # Sequential node for this university (allows short-circuit if needed)
    uni_node = evaluator.add_sequential(
        id=f"University_{index+1}",
        desc=f"{['First','Second','Third','Fourth'][index]} qualifying university identified and verified",
        parent=root_parent,
        critical=False,  # Allow partial scoring across different universities
    )

    # Critical parallel criteria node (all children must pass)
    criteria_node = evaluator.add_parallel(
        id=f"University_{index+1}_Criteria",
        desc=f"Verification that University {index+1} meets all required criteria",
        parent=uni_node,
        critical=True,
    )

    safe_uni_name = uni.university_name or f"University {index+1}"

    # ------------------------- US Location -------------------------------- #
    us_loc_leaf = evaluator.add_leaf(
        id=f"U{index+1}_US_Location",
        desc=f"University {index+1} is located in the United States",
        parent=criteria_node,
        critical=True,
    )
    us_sources = dedup_urls(
        [uni.university_homepage_url, uni.phd_program_url, uni.research_center_url] + (uni.extra_support_urls or [])
    )
    await evaluator.verify(
        claim=f"The university '{safe_uni_name}' is located in the United States.",
        node=us_loc_leaf,
        sources=us_sources,
        additional_instruction=(
            "Look for evidence on the provided webpages that the institution is a U.S. university. "
            "Accept clear indications such as a U.S. campus address, .edu domains, mentions of U.S. states, "
            "or explicit statements identifying the institution as located in the U.S."
        ),
    )

    # ------------------------- PhD Program group --------------------------- #
    phd_group = evaluator.add_parallel(
        id=f"U{index+1}_PhD_Program",
        desc="Doctoral program verification",
        parent=criteria_node,
        critical=True,
    )

    # Existence (custom)
    phd_exists = bool((uni.phd_program_name or "").strip()) and bool((uni.phd_program_url or "").strip())
    evaluator.add_custom_node(
        result=phd_exists,
        id=f"U{index+1}_PhD_Existence",
        desc="Doctoral program exists and is properly identified",
        parent=phd_group,
        critical=True,
    )

    # Relevance
    phd_rel_leaf = evaluator.add_leaf(
        id=f"U{index+1}_PhD_Relevance",
        desc="Doctoral program is in a relevant field as specified in criteria",
        parent=phd_group,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The doctoral program '{uni.phd_program_name or 'the doctoral program'}' is in atmospheric sciences, "
            f"space physics, astrophysics, geophysics, or a directly related interdisciplinary field."
        ),
        node=phd_rel_leaf,
        sources=uni.phd_program_url,
        additional_instruction="Check the program description for field keywords or closely aligned terms.",
    )

    # Currently Active (2024-2026)
    phd_active_leaf = evaluator.add_leaf(
        id=f"U{index+1}_PhD_Currently_Active",
        desc="Program is currently accepting applications or has accepted students within 2024-2026",
        parent=phd_group,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The doctoral program is currently accepting applications or has accepted students within 2024–2026."
        ),
        node=phd_active_leaf,
        sources=dedup_urls([uni.phd_program_url] + (uni.extra_support_urls or [])),
        additional_instruction=(
            "Look for admissions pages, application deadlines, or cohort information indicating 2024, 2025, or 2026. "
            "Mentions like 'Applications open', 'Admission for Fall 2025', or recent cohorts suffice."
        ),
    )

    # Program URL validity
    phd_url_leaf = evaluator.add_leaf(
        id=f"U{index+1}_PhD_URL",
        desc="Valid URL reference provided for the doctoral program",
        parent=phd_group,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"This URL is the official doctoral program page for "
            f"'{uni.phd_program_name or 'the doctoral program'}' and provides program details."
        ),
        node=phd_url_leaf,
        sources=uni.phd_program_url,
        additional_instruction="The page should clearly indicate a PhD or doctoral program with details.",
    )

    # ------------------------- Research Center group ---------------------- #
    center_group = evaluator.add_parallel(
        id=f"U{index+1}_Research_Center",
        desc="Research center/institute/laboratory verification",
        parent=criteria_node,
        critical=True,
    )

    # Center existence (custom)
    center_exists = bool((uni.research_center_name or "").strip()) and bool((uni.research_center_url or "").strip())
    evaluator.add_custom_node(
        result=center_exists,
        id=f"U{index+1}_Center_Existence",
        desc="Research center/institute/laboratory exists and is properly identified",
        parent=center_group,
        critical=True,
    )

    # Center research focus
    center_focus_leaf = evaluator.add_leaf(
        id=f"U{index+1}_Center_Research_Focus",
        desc="Center conducts research in space weather, ionospheric physics, upper atmospheric studies, magnetospheric physics, or stratospheric dynamics",
        parent=center_group,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The research center '{uni.research_center_name or 'the center'}' conducts research in space weather or "
            "related areas such as ionospheric physics, upper atmospheric studies, magnetospheric physics, or "
            "stratospheric dynamics."
        ),
        node=center_focus_leaf,
        sources=uni.research_center_url,
        additional_instruction="Look for explicit keywords on the page indicating these research areas.",
    )

    # Center website publicly accessible describing research
    center_website_leaf = evaluator.add_leaf(
        id=f"U{index+1}_Center_Website",
        desc="Center has publicly accessible website describing research activities",
        parent=center_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The center's website is publicly accessible and describes its research activities.",
        node=center_website_leaf,
        sources=uni.research_center_url,
        additional_instruction="A 'Research', 'Projects', or 'About' section describing activities should be present.",
    )

    # Center URL validity
    center_url_leaf = evaluator.add_leaf(
        id=f"U{index+1}_Center_URL",
        desc="Valid URL reference provided for the research center website",
        parent=center_group,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"This URL corresponds to the research center '{uni.research_center_name or 'the center'}' and presents "
            "official information about it."
        ),
        node=center_url_leaf,
        sources=uni.research_center_url,
        additional_instruction="The page should identify the center by name and affiliation.",
    )

    # ------------------------- NSF Eligibility --------------------------- #
    nsf_leaf = evaluator.add_leaf(
        id=f"U{index+1}_NSF_Eligibility",
        desc=f"University {index+1} is eligible to receive research grants from the U.S. National Science Foundation",
        parent=criteria_node,
        critical=True,
    )
    nsf_sources = dedup_urls(
        [uni.nsf_eligibility_url, uni.research_center_url, uni.university_homepage_url] + (uni.extra_support_urls or [])
    )
    await evaluator.verify(
        claim=(
            f"The university '{safe_uni_name}' is eligible to receive NSF research grants (evidenced by NSF awards or "
            "explicit statements)."
        ),
        node=nsf_leaf,
        sources=nsf_sources,
        additional_instruction="NSF award pages or institutional pages noting NSF-funded projects suffice as proof.",
    )

    # ------------------------- Research Activities ----------------------- #
    ra_group = evaluator.add_parallel(
        id=f"U{index+1}_Research_Activities",
        desc="Research activities verification",
        parent=criteria_node,
        critical=True,
    )

    ra_valid_leaf = evaluator.add_leaf(
        id=f"U{index+1}_Research_Area_Valid",
        desc="Research activities are in space weather prediction, geomagnetic storm studies, ionospheric research, polar vortex dynamics, or stratospheric warming",
        parent=ra_group,
        critical=True,
    )
    ra_sources = dedup_urls([uni.research_center_url] + (uni.extra_support_urls or []))
    await evaluator.verify(
        claim=(
            "The university (via its relevant center or unit) has research activities in at least one of: space weather "
            "prediction, geomagnetic storm studies, ionospheric research, polar vortex dynamics, or stratospheric warming."
        ),
        node=ra_valid_leaf,
        sources=ra_sources,
        additional_instruction="Look for explicit mentions of the listed topics on the provided web pages.",
    )

    ra_doc_leaf = evaluator.add_leaf(
        id=f"U{index+1}_Research_Documented",
        desc="Research activities are documented and verifiable",
        parent=ra_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The aforementioned research activities are publicly documented on the provided webpages.",
        node=ra_doc_leaf,
        sources=ra_sources,
        additional_instruction="Confirm that the page provides concrete descriptions of research topics or projects.",
    )

    # ------------------------- Publications (2023-2026) ------------------ #
    pub_group = evaluator.add_parallel(
        id=f"U{index+1}_Publications",
        desc="Publication verification (2023–2026, relevant field, peer-reviewed)",
        parent=criteria_node,
        critical=True,
    )

    pub_exists_leaf = evaluator.add_leaf(
        id=f"U{index+1}_Publication_Exists",
        desc="At least one qualifying publication is identified",
        parent=pub_group,
        critical=True,
    )
    pub_exists_claim = (
        f"A publication by researchers at {safe_uni_name} is present on this page"
        + (f", titled '{uni.publication_title}'" if (uni.publication_title or "").strip() else "")
        + "."
    )
    await evaluator.verify(
        claim=pub_exists_claim,
        node=pub_exists_leaf,
        sources=uni.publication_url,
        additional_instruction="The page should clearly show a publication record with a title and authors/affiliations.",
    )

    pub_time_leaf = evaluator.add_leaf(
        id=f"U{index+1}_Publication_Timeframe",
        desc="Publication is from 2023-2026",
        parent=pub_group,
        critical=True,
    )
    await evaluator.verify(
        claim="This publication has a publication year in the range 2023–2026 inclusive.",
        node=pub_time_leaf,
        sources=uni.publication_url,
        additional_instruction=(
            f"If a year is listed, check that it is within 2023–2026."
            + (f" The provided year is '{uni.publication_year}'." if uni.publication_year else "")
        ),
    )

    pub_rel_leaf = evaluator.add_leaf(
        id=f"U{index+1}_Publication_Relevance",
        desc="Publication is in atmospheric science, space physics, or related fields",
        parent=pub_group,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The publication is in atmospheric science, space physics, geophysics, astrophysics, or a closely related field."
        ),
        node=pub_rel_leaf,
        sources=uni.publication_url,
        additional_instruction="Use the journal/scope/keywords/abstract to assess relevance.",
    )

    pub_peer_leaf = evaluator.add_leaf(
        id=f"U{index+1}_Publication_Peer_Reviewed",
        desc="Publication is peer-reviewed",
        parent=pub_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The publication is peer-reviewed.",
        node=pub_peer_leaf,
        sources=uni.publication_url,
        additional_instruction=(
            "Journal articles are typically peer-reviewed; many reputable conferences are peer-reviewed as well. "
            "Look for the journal name or reputable publisher; if it's a preprint only, do not count as peer-reviewed."
        ),
    )

    # ------------------------- Collaboration Infrastructure -------------- #
    collab_leaf = evaluator.add_leaf(
        id=f"U{index+1}_Collaboration_Infrastructure",
        desc="University has established policies or infrastructure supporting multi-institutional research collaborations",
        parent=criteria_node,
        critical=True,
    )
    collab_sources = dedup_urls([uni.collaboration_url, uni.research_center_url] + (uni.extra_support_urls or []))
    await evaluator.verify(
        claim=(
            "The university has policies or infrastructure supporting multi-institution collaborations "
            "(e.g., partnership policies, consortia participation, sponsored research agreements, center memberships)."
        ),
        node=collab_leaf,
        sources=collab_sources,
        additional_instruction="Look for explicit mentions of inter-institutional collaborations, consortia, or policies.",
    )

    # ------------------------- Faculty Active ---------------------------- #
    faculty_leaf = evaluator.add_leaf(
        id=f"U{index+1}_Faculty",
        desc="University has documented faculty members actively conducting research in relevant fields",
        parent=criteria_node,
        critical=True,
    )
    faculty_sources = dedup_urls([uni.faculty_url, uni.research_center_url, uni.phd_program_url] + (uni.extra_support_urls or []))
    await evaluator.verify(
        claim=(
            "The university has faculty members actively conducting research in atmospheric science, space physics, "
            "geophysics, astrophysics, or closely related fields."
        ),
        node=faculty_leaf,
        sources=faculty_sources,
        additional_instruction="A faculty listing or people page with research descriptions suffices.",
    )

    # ------------------------- Distinctness ------------------------------ #
    is_distinct = (uni.university_name or "").strip() != "" and (uni.university_name not in prior_university_names)
    evaluator.add_custom_node(
        result=is_distinct,
        id=f"U{index+1}_Distinct",
        desc=f"University {index+1} is distinct from other universities identified in the solution",
        parent=criteria_node,
        critical=True,
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 'space_weather_consortium_universities' task.
    """
    # Initialize evaluator; root is non-critical by design in the framework
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel: 4 universities evaluated independently
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Normalize number of universities: at most 4, pad if fewer
    universities = list(extracted.universities[:4])
    while len(universities) < 4:
        universities.append(UniversityEntry())

    # Add a quick summary info
    evaluator.add_custom_info(
        info={
            "num_universities_parsed": sum(1 for u in universities if (u.university_name or "").strip()),
            "university_names": [u.university_name for u in universities],
        },
        info_type="extraction_overview",
    )

    # Build verification subtrees for each university
    prior_names: List[str] = []
    for idx, uni in enumerate(universities):
        await verify_university(evaluator, root, uni, idx, prior_names)
        # Update seen names for distinctness checks of subsequent entries
        if (uni.university_name or "").strip():
            prior_names.append(uni.university_name)

    # Return summarized evaluation result
    return evaluator.get_summary()