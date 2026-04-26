import asyncio
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ent_projects_2025_2026"
TASK_DESCRIPTION = """
I am researching entertainment projects released between July 2025 and May 2026 that feature actors who also appeared in theatrical films released in 2025. Identify four entertainment projects (films, streaming series, or Broadway shows) that meet all of the following criteria:

1. The project premiered or was released between July 1, 2025, and May 31, 2026.
2. The project involves at least one actor who appeared in a theatrical film released in 2025.
3. Each project must be from a different month within the specified time range.

For each project, provide:
- The project title
- The actor's name who connects it to a 2025 theatrical film
- The actor's role or character name in the project
- The exact premiere/release date
- The platform (for streaming series: Netflix or Hulu) or venue (for Broadway shows: specific theatre name)
- A reference URL from an official source (streaming platform, Broadway League, production company website, or major entertainment news outlet)
"""

DATE_RANGE_START = date(2025, 7, 1)
DATE_RANGE_END = date(2026, 5, 31)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProjectItem(BaseModel):
    title: Optional[str] = None
    project_type: Optional[str] = None  # expected: theatrical film | streaming series | broadway show
    platform_or_venue: Optional[str] = None  # Netflix/Hulu for streaming; theatre name for Broadway; optional for films
    date: Optional[str] = None  # exact day required; any reasonable human format
    date_type_label: Optional[str] = None  # e.g., "premiere", "theatrical release", "official opening"
    actor_name: Optional[str] = None
    actor_role: Optional[str] = None
    project_sources: List[str] = Field(default_factory=list)  # sources for the project info
    actor_2025_film_title: Optional[str] = None  # optional but helpful
    actor_eligibility_sources: List[str] = Field(default_factory=list)  # sources verifying 2025 theatrical film eligibility


class ProjectsExtraction(BaseModel):
    projects: List[ProjectItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_projects() -> str:
    return """
    Extract up to four entertainment projects listed in the answer that meet the task intent. For each project, extract the following fields exactly as they appear in the answer (do not invent anything):

    Required fields per project:
    - title: Project title.
    - project_type: One of exactly "theatrical film", "streaming series", or "broadway show". If the answer uses a synonym (e.g., TV series, limited series, Broadway musical/play), normalize to "streaming series" or "broadway show" accordingly; if it is a movie released in theaters, normalize to "theatrical film".
    - platform_or_venue: If project_type is "streaming series", this must be "Netflix" or "Hulu" (normalize capitalization). If project_type is "broadway show", provide the theatre venue name (as written in the answer, e.g., "Shubert Theatre"). If project_type is "theatrical film", set this to null.
    - date: The exact premiere/release/opening date (include month, day, and year as presented; do not leave only month-year).
    - date_type_label: The label used for the date (e.g., "premiere", "theatrical release", "release", "official opening", "opening", "series premiere").
    - actor_name: The name of one actor who appears in the project and who also appeared in a 2025 theatrical film.
    - actor_role: That actor’s role/character name in this project (as provided in the answer).
    - project_sources: A list of one or more URLs cited for this project; where possible include at least one from an allowed category (official streaming platform page, Broadway League/IBDB, official production company site, or a major entertainment news outlet).
    - actor_2025_film_title: The title of a 2025 theatrical film that the actor appeared in (if provided in the answer).
    - actor_eligibility_sources: A list of URLs (if provided) that verify that this actor appeared in a theatrical film released in 2025 (e.g., major entertainment news outlets, Box Office Mojo, studio sites, etc.).

    Additional rules:
    - If the answer lists more than four candidate projects, extract the first four distinct projects in the order presented.
    - If any field is missing for a project, set it to null or an empty list accordingly.
    - Keep URLs exactly as written; include full protocol. Ignore obviously malformed URLs.
    - Do not add information not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _try_parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    # Attempt parsing with common formats
    from datetime import datetime as dt
    patterns = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%b %d, %Y", "%B %d, %Y",
        "%d %b %Y", "%d %B %Y",
        "%b %-d, %Y", "%B %-d, %Y",  # for UNIX-like day format without zero-padding
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%B %d %Y", "%b %d %Y",
        "%d %B, %Y", "%d %b, %Y",
    ]
    for p in patterns:
        try:
            return dt.strptime(date_str.strip(), p).date()
        except Exception:
            continue
    # Fallback: very loose parsing using month name + day + year via manual heuristic
    import re
    m = re.search(r"(?i)\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                  r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
                  r"[ ,.-]+(\d{1,2})[ ,.-]+(20\d{2})\b", date_str or "")
    if m:
        month_name, day, year = m.group(1), int(m.group(2)), int(m.group(3))
        try:
            month_num = {
                "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
            }[month_name.lower()]
            return date(year, month_num, day)
        except Exception:
            return None
    return None


def _normalize_project_type(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = s.strip().lower()
    # Normalize common synonyms
    if "broadway" in t or ("musical" in t and "broadway" in t) or ("play" in t and "broadway" in t):
        return "broadway show"
    if "series" in t or "season" in t or "tv" in t or "limited series" in t or "miniseries" in t:
        return "streaming series"
    if "film" in t or "movie" in t:
        return "theatrical film"
    # If agent already used one of the canonical strings, keep it
    if t in {"theatrical film", "streaming series", "broadway show"}:
        return t
    return t  # unknown; will likely fail the strict check later


def _expected_date_label_for_type(project_type: Optional[str]) -> List[str]:
    # Return acceptable labels for "correct date type" per project type
    if not project_type:
        return []
    pt = _normalize_project_type(project_type)
    if pt == "theatrical film":
        return ["theatrical release", "release", "theatrical"]
    if pt == "streaming series":
        return ["premiere", "series premiere", "season premiere", "debut", "release"]
    if pt == "broadway show":
        return ["official opening", "opening", "broadway opening"]
    return []


def _date_in_range(d: Optional[date]) -> bool:
    if d is None:
        return False
    return DATE_RANGE_START <= d <= DATE_RANGE_END


def _platform_ok_for_series(platform_or_venue: Optional[str]) -> bool:
    if not platform_or_venue:
        return False
    return platform_or_venue.strip().lower() in {"netflix", "hulu"}


def _venue_ok_for_broadway(platform_or_venue: Optional[str]) -> bool:
    if not platform_or_venue:
        return False
    candidate = platform_or_venue.strip().lower()
    # Must be more than a generic "broadway"; expect a theatre name (basic heuristic)
    if candidate in {"broadway", "none", "n/a"}:
        return False
    return len(candidate) >= 4


def _distinct_titles(projects: List[ProjectItem]) -> bool:
    titles = [p.title.strip().lower() for p in projects if p.title]
    if len(titles) != 4:
        return False
    return len(set(titles)) == 4


def _four_different_months(projects: List[ProjectItem]) -> bool:
    months: List[Tuple[int, int]] = []
    for p in projects:
        d = _try_parse_date(p.date)
        if d is None:
            return False
        months.append((d.year, d.month))
    return len(set(months)) == 4


def _release_date_and_type_ok(p: ProjectItem) -> bool:
    # exact date must be present and within range
    d = _try_parse_date(p.date)
    if not _date_in_range(d):
        return False
    # "correct date type" rough heuristic: provided label should match acceptable aliases
    labels = [lbl.lower() for lbl in _expected_date_label_for_type(p.project_type)]
    if not labels:
        return False
    provided = (p.date_type_label or "").strip().lower()
    # If no label provided, attempt best-effort pass for films (release often implied)
    if not provided:
        # be stricter for series/broadway; lenient for films
        norm_type = _normalize_project_type(p.project_type)
        if norm_type in {"streaming series", "broadway show"}:
            return False
        return True
    # Accept if provided label contains any acceptable alias token
    return any(alias in provided for alias in labels)


def _type_specific_field_ok(p: ProjectItem) -> bool:
    norm = _normalize_project_type(p.project_type)
    if norm == "streaming series":
        return _platform_ok_for_series(p.platform_or_venue)
    if norm == "broadway show":
        return _venue_ok_for_broadway(p.platform_or_venue)
    if norm == "theatrical film":
        # no platform/venue required
        return True
    return False


# --------------------------------------------------------------------------- #
# Verification helpers (per project)                                          #
# --------------------------------------------------------------------------- #
async def verify_project(
    evaluator: Evaluator,
    parent_node,
    project: ProjectItem,
    idx: int,
) -> None:
    proj_no = idx + 1
    proj_node = evaluator.add_parallel(
        id=f"project_{proj_no}",
        desc=f"Project {proj_no} satisfies all per-project requirements",
        parent=parent_node,
        critical=False
    )

    # 1) Project title is provided (custom check)
    title_ok = bool(project.title and project.title.strip())
    evaluator.add_custom_node(
        result=title_ok,
        id=f"project_{proj_no}_project_title",
        desc="Project title is provided",
        parent=proj_node,
        critical=True
    )

    # 2) Project type is identifiable as one of allowed
    norm_type = _normalize_project_type(project.project_type)
    type_ok = norm_type in {"theatrical film", "streaming series", "broadway show"}
    evaluator.add_custom_node(
        result=type_ok,
        id=f"project_{proj_no}_project_type",
        desc="Project type is identifiable as one of: theatrical film, streaming series, or Broadway show",
        parent=proj_node,
        critical=True
    )

    # 3) Exact date provided, in range, and date type correct (custom check)
    date_ok = _release_date_and_type_ok(project)
    evaluator.add_custom_node(
        result=date_ok,
        id=f"project_{proj_no}_release_date_in_range_and_type_correct",
        desc="An exact premiere/release/opening date is provided and is between Jul 1, 2025 and May 31, 2026 (inclusive), using the correct date type for the project (theatrical release for films; premiere for streaming series; official opening for Broadway)",
        parent=proj_node,
        critical=True
    )

    # 4) Connecting actor named and appears in the project (verify via project sources)
    actor_leaf = evaluator.add_leaf(
        id=f"project_{proj_no}_connecting_actor_named",
        desc="An actor is named who appears in the project and is the stated connector to a 2025 theatrical film",
        parent=proj_node,
        critical=True
    )
    actor_name = project.actor_name or ""
    title = project.title or "the project"
    await evaluator.verify(
        claim=f"The actor {actor_name} appears in the project titled '{title}'.",
        node=actor_leaf,
        sources=project.project_sources,
        additional_instruction="Use the provided page(s) to confirm that the named actor is part of the cast/credits of this project. Accept cast list, credits, or official announcement as sufficient confirmation."
    )

    # 5) Actor 2025 theatrical film eligibility (verify via dedicated sources if provided)
    elig_leaf = evaluator.add_leaf(
        id=f"project_{proj_no}_actor_2025_theatrical_film_eligibility",
        desc="The named connecting actor can be verified as having appeared in at least one theatrical film released in 2025 (enough identifying info is provided to verify this claim)",
        parent=proj_node,
        critical=True
    )
    film_title = (project.actor_2025_film_title or "").strip()
    if film_title:
        elig_claim = f"{actor_name} appeared in a theatrical film released in 2025 titled '{film_title}'."
    else:
        elig_claim = f"{actor_name} appeared in at least one theatrical film released in 2025."
    sources_for_elig = project.actor_eligibility_sources if project.actor_eligibility_sources else None
    await evaluator.verify(
        claim=elig_claim,
        node=elig_leaf,
        sources=sources_for_elig,
        additional_instruction="Determine if the actor is in a film that had a theatrical release in calendar year 2025. Prefer explicit wording indicating theatrical release and a 2025 release date."
    )

    # 6) Actor role/character name provided (custom existence check)
    role_ok = bool(project.actor_role and project.actor_role.strip())
    evaluator.add_custom_node(
        result=role_ok,
        id=f"project_{proj_no}_actor_role_in_project",
        desc="That actor’s role/character name in the project is provided",
        parent=proj_node,
        critical=True
    )

    # 7) Type-conditional platform or venue (custom check)
    type_specific_ok = _type_specific_field_ok(project)
    evaluator.add_custom_node(
        result=type_specific_ok,
        id=f"project_{proj_no}_type_specific_platform_or_venue",
        desc="Type-conditional info is correct: if streaming series, platform is specified and is Netflix or Hulu; if Broadway show, the theatre venue name is specified; if theatrical film, no platform/venue field is required",
        parent=proj_node,
        critical=True
    )

    # 8) Reference URL from allowed source category (verify by URLs)
    ref_src_leaf = evaluator.add_leaf(
        id=f"project_{proj_no}_reference_url_from_allowed_source",
        desc="At least one reference URL is provided from an allowed source category (streaming platform, Broadway League, production company website, or major entertainment news outlet)",
        parent=proj_node,
        critical=True
    )
    await evaluator.verify(
        claim=("This webpage belongs to one of the allowed source categories: an official streaming platform page "
               "(Netflix or Hulu), the Broadway League/IBDB, an official production company website, "
               "or a major entertainment news outlet (e.g., Variety, The Hollywood Reporter, Deadline, TheWrap, "
               "New York Times/Arts)."),
        node=ref_src_leaf,
        sources=project.project_sources,
        additional_instruction=("Judge based on the site identity/branding and URL. Accept if the page is clearly from "
                                "netflix.com, hulu.com, broadwayleague.com, ibdb.com, an official studio/production site, "
                                "or a major entertainment news outlet. If the site appears to be a minor blog, fan wiki, "
                                "random forum, or unverified rumor source, mark as not allowed.")
    )

    # 9) Source corroborates required date
    corr_leaf = evaluator.add_leaf(
        id=f"project_{proj_no}_source_corroborates_required_date",
        desc="Provided source(s) corroborate the stated premiere/release/opening date for the project",
        parent=proj_node,
        critical=True
    )
    # Build expected date-type string for the claim
    expected_labels = _expected_date_label_for_type(project.project_type)
    # choose a readable label
    if _normalize_project_type(project.project_type) == "theatrical film":
        exp_label = "theatrical release date"
    elif _normalize_project_type(project.project_type) == "streaming series":
        exp_label = "premiere date"
    elif _normalize_project_type(project.project_type) == "broadway show":
        exp_label = "official opening date"
    else:
        exp_label = "release/premiere/opening date"

    await evaluator.verify(
        claim=f"The page states that '{title}' has its {exp_label} on {project.date}.",
        node=corr_leaf,
        sources=project.project_sources,
        additional_instruction=("Look for explicit mention of the project's date matching the provided value. "
                                "Accept phrasing like 'premieres on', 'opens on', or 'releases on' that clearly conveys "
                                "the stated date.")
    )

    # 10) Source indicates non-speculative (confirmed/announced)
    nonspec_leaf = evaluator.add_leaf(
        id=f"project_{proj_no}_source_indicates_non_speculative",
        desc="Provided source(s) present the required details as confirmed/announced (not speculative/rumored)",
        parent=proj_node,
        critical=True
    )
    await evaluator.verify(
        claim=("The page presents the project's key details (including date/platform/venue) as officially confirmed or "
               "announced, not as rumor/speculation."),
        node=nonspec_leaf,
        sources=project.project_sources,
        additional_instruction=("Reject if the page uses speculative language like 'rumored', 'reportedly', 'might', "
                                "'expected', 'targeting', without an official announcement. Accept official platform/studio/"
                                "Broadway League pages or major outlets citing official announcements.")
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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

    # Extract structured projects from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_projects(),
        template_class=ProjectsExtraction,
        extraction_name="projects_extraction"
    )

    # Normalize and select exactly first 4 items (pad with empties if fewer)
    projects = list(extracted.projects or [])
    if len(projects) > 4:
        # Keep first 4 distinct by title order
        seen = set()
        filtered: List[ProjectItem] = []
        for p in projects:
            t = (p.title or "").strip().lower()
            if t and t not in seen:
                filtered.append(p)
                seen.add(t)
            if len(filtered) >= 4:
                break
        projects = filtered
    while len(projects) < 4:
        projects.append(ProjectItem())

    # Record requirements as ground truth info (for transparency)
    evaluator.add_ground_truth({
        "date_range_start": DATE_RANGE_START.isoformat(),
        "date_range_end": DATE_RANGE_END.isoformat(),
        "required_project_count": 4,
        "month_uniqueness_required": True
    })

    # Root-level critical checks based on extracted content
    # A) Exactly four distinct projects are provided (by title)
    four_distinct = _distinct_titles(projects)
    evaluator.add_custom_node(
        result=four_distinct,
        id="four_distinct_projects_provided",
        desc="Exactly four distinct entertainment projects are provided (no duplicates)",
        parent=root,
        critical=True
    )

    # B) Four different calendar months (month-year combinations)
    months_ok = _four_different_months(projects)
    evaluator.add_custom_node(
        result=months_ok,
        id="different_months",
        desc="The four projects’ premiere/release/opening dates fall in four different calendar months (month-year combinations are all distinct)",
        parent=root,
        critical=True
    )

    # Per-project verification (parallel under root)
    tasks = []
    for i in range(4):
        tasks.append(verify_project(evaluator, root, projects[i], i))
    await asyncio.gather(*tasks)

    # Return structured evaluation summary
    return evaluator.get_summary()