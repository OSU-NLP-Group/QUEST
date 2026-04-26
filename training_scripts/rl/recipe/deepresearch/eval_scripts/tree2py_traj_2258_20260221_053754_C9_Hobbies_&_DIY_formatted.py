import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wildlife_webcams_2026"
TASK_DESCRIPTION = """You're planning a comprehensive DIY wildlife observation and documentation project for 2026. To ensure reliable and high-quality sources for your educational materials, you need to identify 4 different wildlife webcam projects in North America that meet ALL of the following criteria:

1. Must be operated by either a nonprofit organization or a U.S. government agency
2. Must have had at least one camera operational since 2015 or earlier
3. Must have expanded or upgraded their camera system by adding additional cameras after the initial installation
4. Must provide live streaming (not just recorded highlights or seasonal archives)
5. Must be freely accessible to the public without requiring subscription fees
6. Must focus on observable North American wildlife species
7. The 4 projects you identify must collectively represent at least 3 different types of wildlife (such as birds of prey, bears, marine mammals, aquatic life, etc.)

For each of the 4 webcam projects you identify, provide:
- The name of the operating organization or agency
- The year the first camera was installed (must be 2015 or earlier)
- The year when additional cameras were added or the system was upgraded (must be after initial installation)
- The primary wildlife species observed
- The URL of the official webcam page where the live stream can be accessed
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ProjectItem(BaseModel):
    org_name: Optional[str] = None
    org_type: Optional[str] = None  # e.g., "nonprofit", "501(c)(3)", "U.S. government agency", "state agency"
    initial_year: Optional[str] = None  # Prefer a 4-digit year as a string
    expansion_year: Optional[str] = None  # Prefer a 4-digit year as a string; may be null if not provided
    primary_species: Optional[str] = None
    species_type: Optional[str] = None  # e.g., "birds of prey", "bears", "marine mammals", "aquatic life"
    official_page_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class ProjectsExtraction(BaseModel):
    projects: List[ProjectItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_projects() -> str:
    return """
    Extract up to four (4) wildlife webcam projects mentioned in the answer in the order they appear. For each project, return:
    - org_name: The operating organization or agency name
    - org_type: The type of operator as stated (e.g., "nonprofit", "501(c)(3)", "U.S. government agency", "state agency", "federal agency", "national park service", etc.)
    - initial_year: The year the first camera was installed or became operational; if a range is given, return the earliest year; use a 4-digit year string if possible
    - expansion_year: The year additional cameras were added or the system was upgraded; use a 4-digit year string if possible; return null if not mentioned
    - primary_species: The primary wildlife species featured (e.g., "bald eagles", "brown bears", "sea lions")
    - species_type: A concise high-level wildlife type for the primary species, such as "birds of prey", "bears", "marine mammals", "aquatic life", "land mammals", "seabirds", "raptors". Choose one per project.
    - official_page_url: The URL of the official webcam page where the live stream can be accessed
    - additional_urls: Any other URLs mentioned in the answer that are directly related to that project (e.g., organization "about" pages, blog posts about installation or expansion). Do not duplicate the official_page_url here.

    Rules:
    - Only extract projects explicitly mentioned in the answer. Do not invent any projects.
    - Always include full URLs. Accept links presented as markdown; extract the underlying URL.
    - If a field is not present in the answer, set it to null (or an empty array for additional_urls).
    - If more than four projects are mentioned, include only the first four that appear in the answer.
    - Keep org_type and species_type as concise strings, as stated or reasonably summarized from the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth"][n] if 0 <= n < 4 else f"#{n+1}"


def collect_sources(proj: ProjectItem) -> List[str]:
    urls = []
    if proj.official_page_url and proj.official_page_url.strip():
        urls.append(proj.official_page_url.strip())
    for u in proj.additional_urls:
        if u and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification logic per project                                              #
# --------------------------------------------------------------------------- #
async def verify_project(evaluator: Evaluator, parent_node, project: ProjectItem, idx: int) -> None:
    proj_label = f"Project_{idx+1}"
    proj_desc = f"{_ordinal(idx)} wildlife webcam project meeting all requirements"
    project_node = evaluator.add_parallel(
        id=proj_label,
        desc=proj_desc,
        parent=parent_node,
        critical=False  # Non-critical at top level; allows partial credit across projects
    )

    # Official page existence (create early to serve as a critical precondition for deeper checks)
    official_page_exists = evaluator.add_custom_node(
        result=bool(project.official_page_url and project.official_page_url.strip()),
        id=f"P{idx+1}_Official_Page",
        desc="Official webcam page URL is provided",
        parent=project_node,
        critical=True
    )

    # Organization verification
    org_node = evaluator.add_parallel(
        id=f"P{idx+1}_Organization_Verification",
        desc="Verify the operating organization meets requirements",
        parent=project_node,
        critical=True
    )

    # Organization name provided (existence)
    evaluator.add_custom_node(
        result=bool(project.org_name and project.org_name.strip()),
        id=f"P{idx+1}_Org_Name",
        desc="Operating organization name is provided",
        parent=org_node,
        critical=True
    )

    # Organization type verification (nonprofit or US government)
    org_type_leaf = evaluator.add_leaf(
        id=f"P{idx+1}_Org_Type",
        desc="Organization is either a nonprofit or U.S. government agency",
        parent=org_node,
        critical=True
    )
    org_sources = collect_sources(project)
    org_claim = (
        f"The operating organization '{project.org_name or ''}' is either a nonprofit organization "
        f"(e.g., 501(c)(3)) or a U.S. government agency."
    )
    await evaluator.verify(
        claim=org_claim,
        node=org_type_leaf,
        sources=org_sources if org_sources else None,
        additional_instruction=(
            "Use the provided official webcam page and related organization links to determine operator type. "
            "Accept evidence such as 'nonprofit', '501(c)(3)', 'charity', or clear government affiliation (e.g., .gov domain, "
            "National Park Service, U.S. Fish & Wildlife Service, NOAA, state or municipal agencies). "
            "The streaming platform (e.g., YouTube) alone is not sufficient. If the organization's status is unclear, do not support."
        )
    )

    # Camera timeline
    timeline_node = evaluator.add_parallel(
        id=f"P{idx+1}_Camera_Timeline",
        desc="Verify camera installation and expansion history",
        parent=project_node,
        critical=True
    )

    # Initial year provided (existence)
    evaluator.add_custom_node(
        result=bool(project.initial_year and project.initial_year.strip()),
        id=f"P{idx+1}_Initial_Year_Provided",
        desc="Year of first camera installation is provided",
        parent=timeline_node,
        critical=True
    )

    # Initial installation year ≤ 2015 (with sources)
    init_install_leaf = evaluator.add_leaf(
        id=f"P{idx+1}_Initial_Installation",
        desc="First camera was operational by 2015 or earlier",
        parent=timeline_node,
        critical=True
    )
    init_claim_year = project.initial_year or "an earlier year (≤ 2015)"
    init_claim = (
        f"The first camera for this project was operational by {init_claim_year}, "
        f"which is in or before 2015."
    )
    await evaluator.verify(
        claim=init_claim,
        node=init_install_leaf,
        sources=org_sources if org_sources else None,
        additional_instruction=(
            "Look for explicit references to launch/installation dates such as 'since 2012', 'installed in 2014', or similar. "
            "Support only if the evidence indicates the camera was operational in 2015 or earlier."
        )
    )

    # Expansion year provided (existence)
    evaluator.add_custom_node(
        result=bool(project.expansion_year and project.expansion_year.strip()),
        id=f"P{idx+1}_Expansion_Year_Provided",
        desc="Year of system expansion is provided",
        parent=timeline_node,
        critical=True
    )

    # System expansion (with sources)
    expansion_leaf = evaluator.add_leaf(
        id=f"P{idx+1}_System_Expansion",
        desc="Additional cameras were added or system was upgraded after initial installation",
        parent=timeline_node,
        critical=True
    )
    if project.expansion_year and project.initial_year:
        expansion_claim = (
            f"After the initial installation in {project.initial_year}, the project added additional camera(s) "
            f"or upgraded the system in {project.expansion_year}, resulting in multiple views or improved coverage."
        )
    else:
        expansion_claim = (
            "After the initial installation, the project added one or more cameras or upgraded the camera system, "
            "resulting in multiple views or improved coverage."
        )
    await evaluator.verify(
        claim=expansion_claim,
        node=expansion_leaf,
        sources=org_sources if org_sources else None,
        additional_instruction=(
            "Accept evidence like 'added a second camera', 'multi-cam views', 'new angles', '4K/HD upgrade', or "
            "'expanded camera network', and confirm this occurred after the initial installation date. "
            "If timing relative to the initial installation is unclear or no upgrade/additional cameras are evident, do not support."
        )
    )

    # Streaming verification
    streaming_node = evaluator.add_parallel(
        id=f"P{idx+1}_Streaming",
        desc="Verify streaming capabilities and accessibility",
        parent=project_node,
        critical=True
    )

    # Live streaming available
    live_leaf = evaluator.add_leaf(
        id=f"P{idx+1}_Live_Stream",
        desc="Provides live streaming (not just recorded highlights)",
        parent=streaming_node,
        critical=True
    )
    live_claim = (
        "The official webcam page provides a live video stream (not merely archived videos or highlight clips)."
    )
    await evaluator.verify(
        claim=live_claim,
        node=live_leaf,
        sources=project.official_page_url if (project.official_page_url and project.official_page_url.strip()) else None,
        additional_instruction=(
            "Check for an embedded player or link clearly labeled as 'live'. If the page only provides highlight clips, "
            "past recordings, or seasonal archives without an active live stream, do not support. "
            "If clearly seasonal, it still qualifies as long as it provides live streaming during active seasons."
        )
    )

    # Publicly accessible for free
    public_leaf = evaluator.add_leaf(
        id=f"P{idx+1}_Public_Access",
        desc="Freely accessible to public without subscription",
        parent=streaming_node,
        critical=True
    )
    public_claim = (
        "The live stream is freely accessible to the public without requiring subscription fees."
    )
    await evaluator.verify(
        claim=public_claim,
        node=public_leaf,
        sources=project.official_page_url if (project.official_page_url and project.official_page_url.strip()) else None,
        additional_instruction=(
            "If the live stream is viewable on the page or via a public platform (e.g., YouTube) without a paid subscription, support. "
            "If payment or paid membership is required to access the stream, do not support. "
            "A free optional donation appeal does not violate the requirement."
        )
    )

    # Wildlife focus
    wildlife_node = evaluator.add_parallel(
        id=f"P{idx+1}_Wildlife",
        desc="Verify wildlife focus",
        parent=project_node,
        critical=True
    )

    # Species provided (existence)
    evaluator.add_custom_node(
        result=bool(project.primary_species and project.primary_species.strip()),
        id=f"P{idx+1}_Species",
        desc="Primary wildlife species is identified",
        parent=wildlife_node,
        critical=True
    )

    # North American wildlife check
    na_leaf = evaluator.add_leaf(
        id=f"P{idx+1}_North_American",
        desc="Features North American wildlife",
        parent=wildlife_node,
        critical=True
    )
    na_claim = (
        f"The webcam focuses on '{project.primary_species or ''}', a wildlife species observable in North America, "
        f"and the webcam location is in North America."
    )
    await evaluator.verify(
        claim=na_claim,
        node=na_leaf,
        sources=org_sources if org_sources else None,
        additional_instruction=(
            "Use the provided official/project pages to confirm the featured species and/or location is in North America (U.S., Canada, or Mexico). "
            "If the page clearly shows the site is in North America, that suffices even if the species is cosmopolitan."
        )
    )


# --------------------------------------------------------------------------- #
# Wildlife diversity verification (collective criterion)                      #
# --------------------------------------------------------------------------- #
async def verify_wildlife_diversity(evaluator: Evaluator, parent_node, projects: List[ProjectItem]) -> None:
    diversity_leaf = evaluator.add_leaf(
        id="Wildlife_Diversity",
        desc="The 4 projects collectively represent at least 3 different types of wildlife",
        parent=parent_node,
        critical=True
    )

    # Build a concise summary for LLM logical verification
    entries = []
    for i, p in enumerate(projects[:4]):
        entries.append(f"Project {i+1}: species='{p.primary_species or 'N/A'}', type='{p.species_type or 'N/A'}'")

    diversity_claim = (
        "Based solely on the following four projects and their provided primary species and high-level types, "
        "determine whether there are at least three distinct wildlife types represented across the set.\n"
        + "\n".join(entries)
    )

    await evaluator.verify(
        claim=diversity_claim,
        node=diversity_leaf,
        sources=None,
        additional_instruction=(
            "Judge this as a logical check using the listed species/types only (do not fetch external info). "
            "Aggregate to broad categories (e.g., 'raptors' and 'birds of prey' count as the same type; "
            "'whales' and 'seals' are both 'marine mammals' type). "
            "Answer 'Correct' only if there are at least three distinct high-level wildlife types across the four projects."
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
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

    # Extract up to 4 projects from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_projects(),
        template_class=ProjectsExtraction,
        extraction_name="projects_extraction"
    )

    # Normalize to exactly 4 entries (pad with empty if fewer; truncate if more)
    projects: List[ProjectItem] = list(extraction.projects[:4])
    while len(projects) < 4:
        projects.append(ProjectItem())

    # Build project verification subtrees
    for i in range(4):
        await verify_project(evaluator, root, projects[i], i)

    # Collective diversity check
    await verify_wildlife_diversity(evaluator, root, projects)

    # Return structured evaluation result
    return evaluator.get_summary()