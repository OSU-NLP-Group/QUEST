import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "kids_workshops_feb_2026"
TASK_DESCRIPTION = """
I am planning to take my 6-year-old child to free DIY kids workshops in February 2026. Identify one Lowe's Kids Workshop and one Home Depot Kids Workshop that are scheduled in February 2026 and appropriate for my child's age. For each workshop, provide the following information: (1) The name of the project that will be built, (2) The exact date of the workshop, (3) The time window during which the workshop is held, and (4) A direct link to the official registration or information page for that specific workshop.
"""


class WorkshopInfo(BaseModel):
    project_name: Optional[str] = None
    date: Optional[str] = None
    time_window: Optional[str] = None
    url: Optional[str] = None
    extra_urls: List[str] = Field(default_factory=list)


class WorkshopsExtraction(BaseModel):
    lowes: Optional[WorkshopInfo] = None
    homedepot: Optional[WorkshopInfo] = None


def prompt_extract_workshops() -> str:
    return """
    Extract exactly one Lowe's Kids Workshop and one Home Depot Kids Workshop that the answer claims are scheduled in February 2026.

    For each brand, return these fields:
    - project_name: The title/name of the project to be built (string as written in the answer).
    - date: The exact date of the workshop (string as written, e.g., "Saturday, February 1, 2026" or "Feb 1, 2026").
    - time_window: The time range for the workshop (e.g., "9:00 a.m. – 12:00 p.m.", "10 AM - 1 PM").
    - url: A direct link to the official registration or information page for that specific workshop. This should be a URL of the official brand site (for Lowe's or The Home Depot).
    - extra_urls: Any additional URLs the answer cites that are relevant to the brand's kids workshop program (e.g., general program overview page or location-specific info pages). If none, return an empty array.

    Structure the JSON like:
    {
      "lowes": { ... },
      "homedepot": { ... }
    }

    Selection rules:
    - If the answer mentions multiple workshops per brand, select the one scheduled in February 2026. If multiple in February, take the first one mentioned.
    - If the answer does not provide a February 2026 workshop for a brand, set that brand's object to null.
    - Return the fields exactly as presented in the answer text; do not invent missing details.
    - Extract valid URLs only; include full protocol. If a URL is missing protocol, prepend "http://".
    """


def _brand_sources(info: WorkshopInfo) -> List[str]:
    urls: List[str] = []
    if info and info.url:
        urls.append(info.url)
    if info and info.extra_urls:
        urls.extend([u for u in info.extra_urls if isinstance(u, str) and u.strip()])
    return urls


async def _verify_brand(
    evaluator: Evaluator,
    parent_node,
    brand_key: str,
    brand_title: str,
    info: WorkshopInfo,
) -> None:
    """
    Build verification sub-tree for a single brand (Lowe's or Home Depot).
    brand_key: "lowes" or "homedepot"
    brand_title: human-readable brand title for descriptions (e.g., "Lowe's", "Home Depot")
    info: extracted WorkshopInfo for this brand
    """
    # Brand-level sequential node (to gate information by eligibility)
    brand_node = evaluator.add_sequential(
        id=f"{brand_key}_workshop",
        desc=f"{brand_title} Kids Workshop information for February 2026",
        parent=evaluator.root,
        critical=False,
    )

    sources_list = _brand_sources(info)

    # ---------------- Eligibility (Critical) ----------------
    # We split eligibility into two distinct critical leaves: month/year and age suitability.
    eligibility_node = evaluator.add_parallel(
        id=f"{brand_key}_eligibility",
        desc=(
            f"The identified {brand_title} workshop is scheduled in February 2026 and is "
            "age-appropriate for a 6-year-old child (based on official page guidance)"
        ),
        parent=brand_node,
        critical=True,
    )

    # Leaf: scheduled in February 2026 (source-grounded)
    feb_check_node = evaluator.add_leaf(
        id=f"{brand_key}_eligibility_feb_2026",
        desc=f"{brand_title}: Workshop occurs in February 2026 according to the official page",
        parent=eligibility_node,
        critical=True,
    )
    feb_claim = (
        "The official workshop page shows that the event date falls in February 2026."
    )
    await evaluator.verify(
        claim=feb_claim,
        node=feb_check_node,
        sources=info.url,
        additional_instruction="Look for the date on the official page and judge supported only if the month is February and the year is 2026.",
    )

    # Leaf: age appropriateness for a 6-year-old (source-grounded)
    age_check_node = evaluator.add_leaf(
        id=f"{brand_key}_eligibility_age_6_ok",
        desc=f"{brand_title}: A 6-year-old is eligible per the age guidance shown on the official page",
        parent=eligibility_node,
        critical=True,
    )
    age_claim = (
        "According to the official page, the kids workshop age guidance includes age 6 "
        "(e.g., the page states an age range that covers age 6)."
    )
    await evaluator.verify(
        claim=age_claim,
        node=age_check_node,
        sources=sources_list if sources_list else info.url,
        additional_instruction=(
            "Judge supported only if the official page explicitly indicates an age range or guidance that includes age 6 "
            "(for example, 'ages 4–11' or 'ages 5–12' or similar language). If the page provides no age information, return not supported."
        ),
    )

    # ---------------- Information (Non-Critical) ----------------
    info_main = evaluator.add_parallel(
        id=f"{brand_key}_information",
        desc=f"Complete and accurate information about the {brand_title} workshop",
        parent=brand_node,
        critical=False,
    )

    # Identification (Critical group): project name + exact date
    ident_node = evaluator.add_parallel(
        id=f"{brand_key}_identification",
        desc=f"Project name and exact date are provided and accurate according to the official {brand_title} schedule",
        parent=info_main,
        critical=True,
    )

    # Presence checks (critical)
    evaluator.add_custom_node(
        result=bool(info and info.project_name and info.project_name.strip()),
        id=f"{brand_key}_project_name_present",
        desc=f"{brand_title}: Project name is provided in the answer",
        parent=ident_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(info and info.date and info.date.strip()),
        id=f"{brand_key}_date_present",
        desc=f"{brand_title}: Exact date is provided in the answer",
        parent=ident_node,
        critical=True,
    )

    # Verify project name against official page
    proj_leaf = evaluator.add_leaf(
        id=f"{brand_key}_project_name_correct",
        desc=f"{brand_title}: The project name matches the official page",
        parent=ident_node,
        critical=True,
    )
    proj_claim = f"The official page lists the project name as '{info.project_name}'."
    await evaluator.verify(
        claim=proj_claim,
        node=proj_leaf,
        sources=sources_list if sources_list else info.url,
        additional_instruction=(
            "Allow minor formatting or punctuation variants. Judge supported only if the page clearly lists an equivalent project name."
        ),
    )

    # Verify exact date against official page
    date_leaf = evaluator.add_leaf(
        id=f"{brand_key}_date_correct",
        desc=f"{brand_title}: The workshop date matches the official page",
        parent=ident_node,
        critical=True,
    )
    date_claim = f"The official page shows the workshop date as '{info.date}'."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=sources_list if sources_list else info.url,
        additional_instruction=(
            "Be tolerant of reasonable formatting differences (e.g., 'Sat, Feb 1, 2026' vs 'Saturday, February 1, 2026'). "
            "Judge supported only if the same calendar date is clearly indicated on the official page."
        ),
    )

    # Logistics (Critical group): time window + official registration/info link
    log_node = evaluator.add_parallel(
        id=f"{brand_key}_logistics",
        desc=f"Time window and registration/information link to the official {brand_title} page are provided and accurate",
        parent=info_main,
        critical=True,
    )

    # Presence checks (critical)
    evaluator.add_custom_node(
        result=bool(info and info.time_window and info.time_window.strip()),
        id=f"{brand_key}_time_present",
        desc=f"{brand_title}: Time window is provided in the answer",
        parent=log_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(info and info.url and info.url.strip()),
        id=f"{brand_key}_url_present",
        desc=f"{brand_title}: Official registration/information URL is provided in the answer",
        parent=log_node,
        critical=True,
    )

    # Verify time window against official page
    time_leaf = evaluator.add_leaf(
        id=f"{brand_key}_time_window_correct",
        desc=f"{brand_title}: The workshop time window matches the official page",
        parent=log_node,
        critical=True,
    )
    time_claim = f"The official page states the workshop time window is '{info.time_window}'."
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        sources=sources_list if sources_list else info.url,
        additional_instruction=(
            "Allow small formatting variations (e.g., '9am-12pm' vs '9:00 AM – 12:00 PM'). "
            "If the page lists a standard program window (e.g., '9–12') for that date, treat it as supported."
        ),
    )

    # Verify that the provided link is the official registration/information page for the specific workshop
    link_leaf = evaluator.add_leaf(
        id=f"{brand_key}_link_official",
        desc=f"{brand_title}: The provided URL is the official registration or information page for this specific workshop",
        parent=log_node,
        critical=True,
    )
    link_claim = (
        f"The provided URL is an official {brand_title} page (not a third-party site) and specifically the registration/information page "
        f"for the kids workshop '{info.project_name}' scheduled on {info.date}."
    )
    await evaluator.verify(
        claim=link_claim,
        node=link_leaf,
        sources=info.url,
        additional_instruction=(
            "Judge supported only if the URL belongs to the official brand domain and the page content clearly corresponds to this specific kids workshop, "
            "including project name and date/time details. If the URL leads to unrelated or third-party content, return not supported."
        ),
    )


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
    Entry point to evaluate the agent's answer for the kids workshops in February 2026.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Complete information for both Lowe's and Home Depot workshops in February 2026 suitable for a 6-year-old child",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    extracted = await evaluator.extract(
        prompt=prompt_extract_workshops(),
        template_class=WorkshopsExtraction,
        extraction_name="workshops_extraction",
    )

    # Build brand subtrees
    lowes_info = extracted.lowes or WorkshopInfo()
    homedepot_info = extracted.homedepot or WorkshopInfo()

    await _verify_brand(
        evaluator=evaluator,
        parent_node=root,
        brand_key="lowes",
        brand_title="Lowe's",
        info=lowes_info,
    )

    await _verify_brand(
        evaluator=evaluator,
        parent_node=root,
        brand_key="homedepot",
        brand_title="Home Depot",
        info=homedepot_info,
    )

    return evaluator.get_summary()