import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "performers_2025_nov_dec"
TASK_DESCRIPTION = (
    "Identify three performers who appeared in major entertainment releases (including film theatrical/streaming premieres, "
    "TV season finales, or competition show finales) that premiered, were released, or concluded in November or December 2025. "
    "For each performer, provide:\n\n"
    "1. 2025 Production Information:\n"
    "   - The title of the 2025 production\n"
    "   - The performer's role, character name, or competition placement/result\n"
    "   - The exact premiere, release, or finale date in November or December 2025\n"
    "   - Key production details such as: director/creator name, runtime or episode count, season number, or trophy/award name\n"
    "   - A reference URL verifying the production and the performer's participation\n\n"
    "2. Previous Project (2019-2024):\n"
    "   - The title of at least one film or TV project the performer appeared in between 2019 and 2024\n"
    "   - Their role or character name in that project\n"
    "   - The release year of that project\n"
    "   - A reference URL verifying their participation\n\n"
    "3. Performer Professional Background:\n"
    "   - A brief description of the performer's primary profession or notable professional identity (beyond just 'actor')\n"
    "   - A reference URL supporting this background information\n\n"
    "All three performers must have participated in distinct 2025 productions (not the same production). "
    "All information must be verifiable through publicly accessible reference URLs."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Production2025(BaseModel):
    title: Optional[str] = None
    category: Optional[str] = None  # e.g., "film theatrical premiere", "streaming premiere", "TV season finale", "competition show finale"
    event_date: Optional[str] = None  # exact date string in Nov/Dec 2025
    role_or_result: Optional[str] = None
    director_creator: Optional[str] = None
    runtime_episode_count: Optional[str] = None
    season_number: Optional[str] = None
    trophy_award_name: Optional[str] = None
    urls_production_identity: List[str] = Field(default_factory=list)
    urls_event_date: List[str] = Field(default_factory=list)
    urls_participation: List[str] = Field(default_factory=list)
    urls_key_detail: List[str] = Field(default_factory=list)


class PreviousProject(BaseModel):
    title: Optional[str] = None
    year: Optional[str] = None  # keep as string to be flexible, we'll parse
    role: Optional[str] = None
    urls_participation: List[str] = Field(default_factory=list)
    urls_year: List[str] = Field(default_factory=list)


class BackgroundInfo(BaseModel):
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PerformerItem(BaseModel):
    name: Optional[str] = None
    production_2025: Optional[Production2025] = None
    previous_project: Optional[PreviousProject] = None
    background: Optional[BackgroundInfo] = None


class PerformersExtraction(BaseModel):
    performers: List[PerformerItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_performers() -> str:
    return (
        "Extract up to three performers described in the answer. For each performer, return a JSON object with fields:\n\n"
        "name: The performer's full name.\n"
        "production_2025: {\n"
        "  title: Exact title of the 2025 production;\n"
        "  category: The event type among these: 'film theatrical premiere', 'streaming premiere', 'TV season finale', or 'competition show finale' "
        "(use a close phrase if the answer uses a slightly different wording);\n"
        "  event_date: The exact premiere/release/finale date string in November or December 2025 as written in the answer;\n"
        "  role_or_result: The performer's role/character name or competition placement/result;\n"
        "  director_creator: A director or creator name if provided;\n"
        "  runtime_episode_count: A runtime or episode count if provided;\n"
        "  season_number: A season number if provided;\n"
        "  trophy_award_name: A trophy/award name if provided;\n"
        "  urls_production_identity: URL(s) cited that verify the production identity (title);\n"
        "  urls_event_date: URL(s) cited that verify the Nov/Dec 2025 event date;\n"
        "  urls_participation: URL(s) cited that verify the performer's participation and role/result in the 2025 production;\n"
        "  urls_key_detail: URL(s) cited that verify the provided key production detail (director/creator, runtime/episodes, season, or trophy/award).\n"
        "}\n\n"
        "previous_project: {\n"
        "  title: Title of at least one film or TV project between 2019 and 2024;\n"
        "  year: Release year between 2019 and 2024 (inclusive) as written in the answer;\n"
        "  role: The performer's role/character name in that project;\n"
        "  urls_participation: URL(s) cited that verify their participation/role in the previous project;\n"
        "  urls_year: URL(s) cited that verify the previous project's release year.\n"
        "}\n\n"
        "background: {\n"
        "  description: A brief professional background/primary professional identity for the performer that is more specific than just 'actor';\n"
        "  urls: URL(s) that verify this background.\n"
        "}\n\n"
        "Return a top-level JSON object with a 'performers' array containing up to 3 such performer objects. "
        "If some fields are missing in the answer, set them to null or an empty list accordingly. Extract only URLs explicitly present."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_title(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    return re.sub(r"\s+", " ", t).strip().lower()


def parse_year_from_string(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    # Extract first 4-digit year occurrence
    m = re.search(r"(20\d{2})", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def is_year_in_range_2019_2024(s: Optional[str]) -> bool:
    y = parse_year_from_string(s)
    return y is not None and 2019 <= y <= 2024


def month_in_nov_dec_2025(date_str: Optional[str]) -> bool:
    if not date_str:
        return False
    s = date_str.strip().lower()
    if "2025" not in s:
        # Also allow YYYY-MM-DD style with 2025
        if not re.search(r"2025", s):
            return False

    # Check textual month names
    if any(m in s for m in ["nov", "november", "dec", "december"]):
        return True

    # Numeric formats: YYYY-MM-DD or MM/DD/YYYY
    # YYYY-MM-DD where MM is 11 or 12
    if re.search(r"2025[-/.](11|12)[-/.]\d{1,2}", s):
        return True
    # MM/DD/YYYY (or MM-DD-YYYY etc.), MM = 11 or 12
    if re.search(r"(11|12)[-/.]\d{1,2}[-/.]2025", s):
        return True

    return False


def category_qualifies(cat: Optional[str]) -> bool:
    if not cat:
        return False
    c = cat.strip().lower()
    # Accept reasonable phrasings
    qualifies = (
        ("premiere" in c and ("film" in c or "movie" in c or "stream" in c)) or
        ("release" in c and ("film" in c or "movie" in c or "stream" in c)) or
        ("season finale" in c) or
        ("finale" in c and ("tv" in c or "season" in c or "competition" in c))
    )
    return qualifies


def has_nonempty(value: Optional[str]) -> bool:
    return bool(value and value.strip())


def has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls and len([u for u in urls if has_nonempty(u)]) > 0)


def pick_key_detail(prod: Optional[Production2025]) -> Tuple[Optional[str], Optional[str]]:
    """
    Pick the first available key production detail and return (detail_type, detail_value).
    detail_type in {"director_creator", "runtime_episode_count", "season_number", "trophy_award_name"}
    """
    if not prod:
        return None, None
    if has_nonempty(prod.director_creator):
        return "director_creator", prod.director_creator
    if has_nonempty(prod.runtime_episode_count):
        return "runtime_episode_count", prod.runtime_episode_count
    if has_nonempty(prod.season_number):
        return "season_number", prod.season_number
    if has_nonempty(prod.trophy_award_name):
        return "trophy_award_name", prod.trophy_award_name
    return None, None


def event_phrase_from_category(cat: Optional[str]) -> str:
    c = (cat or "").strip().lower()
    if "premiere" in c:
        return "premiere"
    if "release" in c:
        return "release"
    if "finale" in c and "season" in c:
        return "season finale"
    if "finale" in c:
        return "finale"
    return "event date"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_production_2025(
    evaluator: Evaluator,
    parent_node,
    perf_name: Optional[str],
    prod: Optional[Production2025],
) -> None:
    node_prod = evaluator.add_parallel(
        id="production_2025",
        desc="2025 qualifying production information for this performer",
        parent=parent_node,
        critical=True
    )

    # Existence/format checks (custom nodes)
    evaluator.add_custom_node(
        result=has_nonempty(prod.title) if prod else False,
        id="production_title_provided",
        desc="Exact 2025 production title is provided",
        parent=node_prod,
        critical=True
    )

    evaluator.add_custom_node(
        result=category_qualifies(prod.category) if prod else False,
        id="production_category_qualifies",
        desc="2025 production is explicitly a film theatrical/streaming premiere OR a TV season finale OR a competition show finale",
        parent=node_prod,
        critical=True
    )

    evaluator.add_custom_node(
        result=month_in_nov_dec_2025(prod.event_date) if prod else False,
        id="exact_event_date_provided_and_in_nov_dec_2025",
        desc="Exact premiere/release/finale date is provided and is in November or December 2025",
        parent=node_prod,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_nonempty(prod.role_or_result) if prod else False,
        id="performer_role_or_result_provided",
        desc="Performer’s role/character name OR competition placement/result in the 2025 production is provided",
        parent=node_prod,
        critical=True
    )

    evaluator.add_custom_node(
        result=(pick_key_detail(prod)[0] is not None) if prod else False,
        id="at_least_one_key_production_detail_provided",
        desc="At least one key production detail is provided (director/creator name OR runtime/episode count OR season number OR trophy/award name)",
        parent=node_prod,
        critical=True
    )

    # URL verifications (leaf nodes)
    # 1) Production identity (title)
    if has_any_url(prod.urls_production_identity if prod else []):
        leaf_title_verify = evaluator.add_leaf(
            id="url_verifies_production_identity",
            desc="At least one publicly accessible URL is provided that supports the 2025 production identity (title)",
            parent=node_prod,
            critical=True
        )
        claim = f"The title of the production is '{prod.title}'."
        await evaluator.verify(
            claim=claim,
            node=leaf_title_verify,
            sources=prod.urls_production_identity,
            additional_instruction=(
                "Verify that the cited source identifies the same production title. Allow minor punctuation/casing variations."
            )
        )
    else:
        evaluator.add_leaf(
            id="url_verifies_production_identity",
            desc="At least one publicly accessible URL is provided that supports the 2025 production identity (title)",
            parent=node_prod,
            critical=True,
            score=0.0,
            status="failed"
        )

    # 2) Event date (Nov/Dec 2025)
    if has_any_url(prod.urls_event_date if prod else []):
        leaf_date_verify = evaluator.add_leaf(
            id="url_verifies_2025_event_date",
            desc="At least one publicly accessible URL is provided that supports the stated Nov/Dec 2025 premiere/release/finale date",
            parent=node_prod,
            critical=True
        )
        event_phrase = event_phrase_from_category(prod.category)
        claim = f"The {event_phrase} date of '{prod.title}' is {prod.event_date}."
        await evaluator.verify(
            claim=claim,
            node=leaf_date_verify,
            sources=prod.urls_event_date,
            additional_instruction=(
                "Confirm that the cited source supports the specific event date in 2025. "
                "Interpret the event as a premiere/release/finale according to the category; "
                "allow reasonable wording variants."
            )
        )
    else:
        evaluator.add_leaf(
            id="url_verifies_2025_event_date",
            desc="At least one publicly accessible URL is provided that supports the stated Nov/Dec 2025 premiere/release/finale date",
            parent=node_prod,
            critical=True,
            score=0.0,
            status="failed"
        )

    # 3) Performer participation and role/result
    if has_any_url(prod.urls_participation if prod else []):
        leaf_participation_verify = evaluator.add_leaf(
            id="url_verifies_performer_participation",
            desc="At least one publicly accessible URL is provided that supports the performer’s participation and stated role/result in the 2025 production",
            parent=node_prod,
            critical=True
        )
        claim = (
            f"{perf_name} participated in '{prod.title}' in 2025 with the role/result '{prod.role_or_result}'."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf_participation_verify,
            sources=prod.urls_participation,
            additional_instruction=(
                "Verify the performer's credit or participation and the role/result. "
                "Allow minor name or role wording variants."
            )
        )
    else:
        evaluator.add_leaf(
            id="url_verifies_performer_participation",
            desc="At least one publicly accessible URL is provided that supports the performer’s participation and stated role/result in the 2025 production",
            parent=node_prod,
            critical=True,
            score=0.0,
            status="failed"
        )

    # 4) Key production detail verification
    detail_type, detail_value = pick_key_detail(prod)
    if has_any_url(prod.urls_key_detail if prod else []) and detail_type and detail_value:
        leaf_keydetail_verify = evaluator.add_leaf(
            id="url_verifies_key_production_detail",
            desc="At least one publicly accessible URL is provided that supports the stated key production detail",
            parent=node_prod,
            critical=True
        )
        # Build claim according to detail_type
        if detail_type == "director_creator":
            claim = f"The director/creator for '{prod.title}' is '{detail_value}'."
        elif detail_type == "runtime_episode_count":
            claim = f"The runtime/episode count for '{prod.title}' is '{detail_value}'."
        elif detail_type == "season_number":
            claim = f"The season number for '{prod.title}' is '{detail_value}'."
        elif detail_type == "trophy_award_name":
            claim = f"The trophy/award name associated with '{prod.title}' is '{detail_value}'."
        else:
            claim = f"A key production detail for '{prod.title}' is '{detail_value}'."

        await evaluator.verify(
            claim=claim,
            node=leaf_keydetail_verify,
            sources=prod.urls_key_detail,
            additional_instruction=(
                "Verify that the cited source explicitly supports this specific key production detail for the same production."
            )
        )
    else:
        evaluator.add_leaf(
            id="url_verifies_key_production_detail",
            desc="At least one publicly accessible URL is provided that supports the stated key production detail",
            parent=node_prod,
            critical=True,
            score=0.0,
            status="failed"
        )


async def verify_previous_project(
    evaluator: Evaluator,
    parent_node,
    perf_name: Optional[str],
    prev: Optional[PreviousProject],
) -> None:
    node_prev = evaluator.add_parallel(
        id="previous_project_2019_2024",
        desc="At least one previous (2019–2024) film/TV project for this performer",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_nonempty(prev.title) if prev else False,
        id="previous_project_title_provided",
        desc="Title of at least one film or TV project is provided",
        parent=node_prev,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_year_in_range_2019_2024(prev.year) if prev else False,
        id="previous_project_year_provided_and_in_range",
        desc="Release year is provided and is between 2019 and 2024 (inclusive)",
        parent=node_prev,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_nonempty(prev.role) if prev else False,
        id="previous_project_role_provided",
        desc="Performer’s role/character name in the previous project is provided",
        parent=node_prev,
        critical=True
    )

    # URL verification for participation
    if has_any_url(prev.urls_participation if prev else []):
        leaf_prev_participation = evaluator.add_leaf(
            id="url_verifies_previous_project_participation",
            desc="At least one publicly accessible URL is provided that supports the performer’s participation/role in the previous project",
            parent=node_prev,
            critical=True
        )
        claim = (
            f"In the previous project '{prev.title}', {perf_name} is credited with the role '{prev.role}'."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf_prev_participation,
            sources=prev.urls_participation,
            additional_instruction=(
                "Verify the performer's participation/role for the stated previous project. "
                "Allow minor wording variants."
            )
        )
    else:
        evaluator.add_leaf(
            id="url_verifies_previous_project_participation",
            desc="At least one publicly accessible URL is provided that supports the performer’s participation/role in the previous project",
            parent=node_prev,
            critical=True,
            score=0.0,
            status="failed"
        )

    # URL verification for release year
    if has_any_url(prev.urls_year if prev else []):
        leaf_prev_year = evaluator.add_leaf(
            id="url_verifies_previous_project_year",
            desc="At least one publicly accessible URL is provided that supports the previous project’s release year",
            parent=node_prev,
            critical=True
        )
        claim = f"The release year of '{prev.title}' is {prev.year}."
        await evaluator.verify(
            claim=claim,
            node=leaf_prev_year,
            sources=prev.urls_year,
            additional_instruction=(
                "Verify that the cited source supports the project's release year in the range 2019–2024."
            )
        )
    else:
        evaluator.add_leaf(
            id="url_verifies_previous_project_year",
            desc="At least one publicly accessible URL is provided that supports the previous project’s release year",
            parent=node_prev,
            critical=True,
            score=0.0,
            status="failed"
        )


async def verify_background(
    evaluator: Evaluator,
    parent_node,
    perf_name: Optional[str],
    bg: Optional[BackgroundInfo],
) -> None:
    node_bg = evaluator.add_parallel(
        id="professional_background",
        desc="Performer professional background / primary professional identity with citation",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_nonempty(bg.description) if bg else False,
        id="background_description_provided",
        desc="A brief professional background/primary professional identity is provided (more specific than only a bare label like 'actor')",
        parent=node_bg,
        critical=True
    )

    if has_any_url(bg.urls if bg else []):
        leaf_bg_verify = evaluator.add_leaf(
            id="url_verifies_background",
            desc="At least one publicly accessible URL is provided that supports the stated professional background",
            parent=node_bg,
            critical=True
        )
        claim = f"The performer {perf_name} has the professional background: {bg.description}."
        await evaluator.verify(
            claim=claim,
            node=leaf_bg_verify,
            sources=bg.urls,
            additional_instruction=(
                "Verify that the cited source supports the stated primary professional identity/background of the performer."
            )
        )
    else:
        evaluator.add_leaf(
            id="url_verifies_background",
            desc="At least one publicly accessible URL is provided that supports the stated professional background",
            parent=node_bg,
            critical=True,
            score=0.0,
            status="failed"
        )


async def verify_performer(
    evaluator: Evaluator,
    root_node,
    performer: PerformerItem,
    performer_index: int
) -> Optional[str]:
    """
    Build tree for a single performer and return the 2025 production title (for distinctness check).
    """
    perf_node = evaluator.add_parallel(
        id=f"performer_{performer_index}",
        desc=f"Performer {performer_index} details (2025 production + prior project + background)",
        parent=root_node,
        critical=False
    )

    # Performer name provided
    evaluator.add_custom_node(
        result=has_nonempty(performer.name),
        id="performer_name_provided",
        desc="Performer’s name is provided",
        parent=perf_node,
        critical=True
    )

    # 2025 production subtree
    await verify_production_2025(evaluator, perf_node, performer.name, performer.production_2025)

    # Previous project subtree
    await verify_previous_project(evaluator, perf_node, performer.name, performer.previous_project)

    # Background subtree
    await verify_background(evaluator, perf_node, performer.name, performer.background)

    # Return title for distinctness check
    return performer.production_2025.title if (performer.production_2025 and performer.production_2025.title) else None


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

    # Extract performers
    extraction = await evaluator.extract(
        prompt=prompt_extract_performers(),
        template_class=PerformersExtraction,
        extraction_name="performers_extraction"
    )

    # Prepare exactly 3 performers (pad if fewer)
    performers_list: List[PerformerItem] = list(extraction.performers or [])
    while len(performers_list) < 3:
        performers_list.append(PerformerItem())

    # Build verification subtrees per performer
    titles_for_distinctness: List[Optional[str]] = []
    for i in range(1, 4):  # 1..3 for performer indices
        title = await verify_performer(evaluator, root, performers_list[i - 1], i)
        titles_for_distinctness.append(title)

    # Distinct productions check (critical)
    normalized_titles = [normalize_title(t) for t in titles_for_distinctness if t]
    distinct_ok = (
        len(normalized_titles) == 3 and
        all(t is not None for t in normalized_titles) and
        len(set(normalized_titles)) == 3
    )

    evaluator.add_custom_node(
        result=distinct_ok,
        id="distinct_2025_productions",
        desc="The three performers’ 2025 production titles are all different (distinct productions)",
        parent=root,
        critical=True
    )

    # Return summary
    return evaluator.get_summary()