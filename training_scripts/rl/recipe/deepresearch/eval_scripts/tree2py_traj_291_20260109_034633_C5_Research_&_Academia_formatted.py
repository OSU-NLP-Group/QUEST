import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "du_centers_2024_25"
TASK_DESCRIPTION = (
    "In 2024/2025, the University of Denver officially designated six research centers for the first time in its history. "
    "Among these newly designated research centers, identify the four centers that focus specifically on the following research areas: "
    "(1) A center that focuses on data and business analytics to help stakeholders make informed decisions, "
    "(2) A center that focuses on consumer behavior and market-driven research, "
    "(3) A center that studies journalism, civic engagement, and emerging digital media environments, and "
    "(4) A center that conducts multidisciplinary research on immigration policy, with a particular focus on Latin American immigrant populations. "
    "For each of the four research centers, provide: the official name of the center, the school or college at the University of Denver where it is housed, "
    "a brief description of its research focus, and a reference URL."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class Center(BaseModel):
    official_name: Optional[str] = None
    housed_school_or_college: Optional[str] = None
    focus_description: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class CentersByCategory(BaseModel):
    data_analytics: Optional[Center] = None
    consumer: Optional[Center] = None
    journalism: Optional[Center] = None
    immigration: Optional[Center] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_centers_by_category() -> str:
    return """
    Extract up to four University of Denver research centers from the answer, assigning one center to each of the following categories:
    - data_analytics: A center focused on data and business analytics, supporting stakeholders' decision-making (public or private sector).
    - consumer: A center focused on consumer behavior and market-driven research.
    - journalism: A center focused on journalism, civic engagement, and emerging digital media environments (for democratic engagement).
    - immigration: A center conducting multidisciplinary immigration policy research with a particular focus on Latin American immigrant populations (including the Rocky Mountain West).

    For each category, return an object with:
    - official_name: the official name of the center as stated in the answer.
    - housed_school_or_college: the DU school or college where the center is housed.
    - focus_description: a brief phrasing of the center’s research focus as presented.
    - reference_urls: an array of the reference URLs explicitly listed in the answer for that center (can be the center’s official page or DU news/announcements). Extract actual URLs only; do not fabricate.

    If the answer provides multiple possible centers for a category, choose the first one that best matches the category.
    If the answer omits a category, set that category field to null.
    If a field is not present for a chosen center, set it to null (or [] for reference_urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _valid_urls(urls: List[str]) -> List[str]:
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u2 = u.strip()
        if not u2:
            continue
        if u2.startswith(("http://", "https://")):
            cleaned.append(u2)
        else:
            # Prepend http:// if protocol missing (per framework guidance)
            cleaned.append("http://" + u2)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for x in cleaned:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _names_list(extracted: CentersByCategory) -> List[str]:
    names = []
    for center in [extracted.data_analytics, extracted.consumer, extracted.journalism, extracted.immigration]:
        if center and _nonempty(center.official_name):
            names.append(center.official_name.strip())
    return names


# --------------------------------------------------------------------------- #
# Category-specific verification                                              #
# --------------------------------------------------------------------------- #
async def verify_center_category(
    evaluator: Evaluator,
    parent_node,
    category_id: str,
    category_desc: str,
    center: Optional[Center],
    focus_constraint_instruction: str,
) -> None:
    """
    Build verification sub-tree for one category with the following leaves:
    - Reference_URL (custom existence/validity check)
    - Official_Name (verify by URL)
    - Entity_Is_Research_Center_Not_Institute (verify by URL)
    - Newly_Designated_2024_2025 (verify by URL)
    - Housed_School_or_College (verify by URL)
    - Focus_Matches_* (verify by URL with category-specific instruction)
    """
    cat_node = evaluator.add_parallel(
        id=f"{category_id}",
        desc=category_desc,
        parent=parent_node,
        critical=False  # Non-critical under root; each category allows partial credit
    )

    # Normalize center and URLs
    name = center.official_name.strip() if (center and _nonempty(center.official_name)) else ""
    housed = center.housed_school_or_college.strip() if (center and _nonempty(center.housed_school_or_college)) else ""
    urls = _valid_urls(center.reference_urls if center else [])

    # 1) Reference_URL — custom existence/format check (critical)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"{category_id}_Reference_URL",
        desc="Provide at least one valid reference URL for the center.",
        parent=cat_node,
        critical=True
    )

    # 2) Official_Name — verify claimed official name appears on provided URLs
    official_leaf = evaluator.add_leaf(
        id=f"{category_id}_Official_Name",
        desc="Provide the official name of the center.",
        parent=cat_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official name of the center is '{name}'.",
        node=official_leaf,
        sources=urls,
        additional_instruction="Confirm that the page(s) clearly show this exact center name or an obviously equivalent official name."
    )

    # 3) Entity_Is_Research_Center_Not_Institute — verify it's a DU research center, not an institute
    type_leaf = evaluator.add_leaf(
        id=f"{category_id}_Entity_Is_Research_Center_Not_Institute",
        desc="Entity is designated as a DU research center (not a research institute).",
        parent=cat_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{name}' is a research center (not an institute) at the University of Denver.",
        node=type_leaf,
        sources=urls,
        additional_instruction="Confirm that the entity is explicitly a research center at DU. If the page refers to it as an institute or something else, this should fail."
    )

    # 4) Newly_Designated_2024_2025 — verify designation timing
    new_leaf = evaluator.add_leaf(
        id=f"{category_id}_Newly_Designated_2024_2025",
        desc="Center was newly designated by DU in the 2024/2025 academic year.",
        parent=cat_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The University of Denver newly designated '{name}' as a research center in the 2024/2025 academic year.",
        node=new_leaf,
        sources=urls,
        additional_instruction="Look for DU announcements or pages stating that in AY 2024–2025 (July 2024–June 2025), DU officially designated six research centers and that this center was one of them. Accept phrasing like '2024-25', '2024/25', or '2024–2025'."
    )

    # 5) Housed_School_or_College — verify housing unit
    housed_leaf = evaluator.add_leaf(
        id=f"{category_id}_Housed_School_or_College",
        desc="Identify the DU school or college where the center is housed.",
        parent=cat_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The center '{name}' is housed in '{housed}' at the University of Denver.",
        node=housed_leaf,
        sources=urls,
        additional_instruction="Accept reasonable synonyms like 'housed in', 'based in', 'within', 'affiliated with', as long as the school/college is correctly identified (e.g., Daniels College of Business, Josef Korbel School of International Studies, etc.)."
    )

    # 6) Focus_Matches_* — verify category-specific focus alignment
    focus_leaf = evaluator.add_leaf(
        id=f"{category_id}_Focus_Matches",
        desc="Brief focus description matches the category-specific constraint.",
        parent=cat_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The center '{name}' has a research focus that aligns with the following description: {focus_constraint_instruction}",
        node=focus_leaf,
        sources=urls,
        additional_instruction="Use the mission/overview text on the provided page(s) to judge alignment. Allow minor paraphrasing; ensure the key elements are explicitly supported."
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
    Evaluate an answer for the DU research centers (2024/2025) task.
    """
    # 1) Initialize evaluator (root is parallel by default in this task)
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

    # 2) Extract four centers by category from the answer
    extracted: CentersByCategory = await evaluator.extract(
        prompt=prompt_extract_centers_by_category(),
        template_class=CentersByCategory,
        extraction_name="centers_by_category"
    )

    # 3) Set Integrity: exactly four distinct centers, one per category (no duplicates)
    names = _names_list(extracted)
    all_four_present = (
        extracted.data_analytics is not None and _nonempty(extracted.data_analytics.official_name) and
        extracted.consumer is not None and _nonempty(extracted.consumer.official_name) and
        extracted.journalism is not None and _nonempty(extracted.journalism.official_name) and
        extracted.immigration is not None and _nonempty(extracted.immigration.official_name)
    )
    distinct_names = len(names) == 4 and len(set([n.lower() for n in names])) == 4
    evaluator.add_custom_node(
        result=all_four_present and distinct_names,
        id="Set_Integrity",
        desc="Response includes exactly four distinct centers, one per specified focus area (no duplicates).",
        parent=root,
        critical=True
    )

    # 4) Category verification subtrees (all parallel under root)
    # Data & Business Analytics
    await verify_center_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="Data_Analytics_Center",
        category_desc="Center matching the data and business analytics focus area.",
        center=extracted.data_analytics,
        focus_constraint_instruction="data and business analytics that help public- and private-sector stakeholders make informed decisions."
    )

    # Consumer behavior / market-driven research
    await verify_center_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="Consumer_Research_Center",
        category_desc="Center matching the consumer behavior and market-driven research focus area.",
        center=extracted.consumer,
        focus_constraint_instruction="study of consumer (human) behavior with emphasis on market-driven research and actionable market insights."
    )

    # Journalism, civic engagement, emerging digital media
    await verify_center_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="Journalism_Center",
        category_desc="Center matching the journalism, civic engagement, and emerging digital media environments focus area.",
        center=extracted.journalism,
        focus_constraint_instruction="journalism’s role in civic or democratic engagement and examination of emergent digital media environments for such engagement."
    )

    # Immigration policy, Latin American focus
    await verify_center_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="Immigration_Center",
        category_desc="Center matching the multidisciplinary immigration policy research focus area (Latin America focus).",
        center=extracted.immigration,
        focus_constraint_instruction="multidisciplinary immigration policy research focusing primarily on immigrant populations from Latin America (including the Rocky Mountain West)."
    )

    # 5) Return evaluation summary
    return evaluator.get_summary()