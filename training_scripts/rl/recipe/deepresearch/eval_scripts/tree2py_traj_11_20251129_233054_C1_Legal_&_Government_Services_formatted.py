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
TASK_ID = "ana_reyes_dismissal"
TASK_DESCRIPTION = """
In late 2025, the U.S. Department of Justice filed a misconduct complaint against U.S. District Judge Ana Reyes in Washington, D.C. regarding her conduct during hearings in a case challenging President Trump's transgender military ban. This complaint was subsequently dismissed by a court official. Identify the full name and position/court of the official who dismissed this complaint, and provide the date of the dismissal order.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DismissalInfo(BaseModel):
    """Information about the dismissal official and dismissal order."""
    official_name: Optional[str] = None
    official_title: Optional[str] = None
    official_court: Optional[str] = None
    dismissal_order_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_dismissal_info() -> str:
    return """
    From the provided answer, extract the following information about the dismissal of the DOJ misconduct complaint against U.S. District Judge Ana Reyes (related to hearings in the challenge to President Trump's transgender military ban) in late 2025:

    1. official_name: The full name of the court official who dismissed the complaint.
    2. official_title: The official’s position/title at the time of dismissal (e.g., chief judge, clerk, etc.).
    3. official_court: The court or office the dismissing official serves (court affiliation).
    4. dismissal_order_date: The calendar date when the dismissal order was issued (as stated in the answer).
    5. sources: An array of all URLs cited in the answer that support these details (include every URL specifically referenced; parse markdown links to capture the actual URL).

    Rules:
    - Extract exactly what is stated in the answer; do not invent or normalize beyond what is present.
    - If any item is not mentioned in the answer, set it to null.
    - For sources, include only URLs explicitly present in the answer. If none are present, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper: Build additional instruction for verification                       #
# --------------------------------------------------------------------------- #
def build_base_instruction(info: DismissalInfo) -> str:
    return (
        "Use the cited source URLs to verify the claim details about the dismissal of the DOJ misconduct complaint "
        "against U.S. District Judge Ana Reyes, connected to hearings in the challenge to President Trump's transgender "
        "military ban, in late 2025. Confirm that the webpage explicitly supports the specific claim being verified. "
        "If the URLs are missing, irrelevant, or inaccessible, or the webpage does not explicitly support the claim, "
        "treat the claim as not supported and mark it incorrect."
    )


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_dismissal_info(
    evaluator: Evaluator,
    parent_node,
    info: DismissalInfo,
) -> None:
    """
    Build the verification subtree for the dismissal information and run checks.
    """
    # Parent node (critical) aggregating all required fields
    group_node = evaluator.add_parallel(
        id="ana_reyes_dismissal_info",
        desc=(
            "Provides complete information about who dismissed the U.S. DOJ misconduct complaint against "
            "U.S. District Judge Ana Reyes (arising from hearings in the transgender military ban challenge) "
            "and when the dismissal order was issued."
        ),
        parent=parent_node,
        critical=True,
    )

    base_ins = build_base_instruction(info)
    srcs = info.sources if info and info.sources else []

    # 1) Dismissing official full name
    name_node = evaluator.add_leaf(
        id="dismissing_official_full_name",
        desc=(
            "Provides the full name of the court official who dismissed the DOJ misconduct complaint against "
            "Judge Ana Reyes (transgender military ban challenge-related hearings)."
        ),
        parent=group_node,
        critical=True,
    )
    name_value = info.official_name or ""
    name_claim = (
        f"The complaint was dismissed by the official named '{name_value}'. "
        f"This is specifically in reference to the DOJ misconduct complaint regarding Judge Ana Reyes."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=srcs,
        additional_instruction=(
            base_ins
            + " If the name in the claim is missing or empty, consider the claim incorrect."
        ),
    )

    # 2) Dismissing official title/position
    title_node = evaluator.add_leaf(
        id="dismissing_official_title",
        desc="Provides the dismissing official’s position/title (e.g., chief judge, clerk, etc.) at the time of dismissal.",
        parent=group_node,
        critical=True,
    )
    title_value = info.official_title or ""
    title_claim = (
        f"The dismissing official held the position/title '{title_value}' at the time of dismissal."
    )
    await evaluator.verify(
        claim=title_claim,
        node=title_node,
        sources=srcs,
        additional_instruction=(
            base_ins
            + " Verify the official’s role/title on the cited webpage(s). Minor variations in phrasing are acceptable "
              "(e.g., 'Chief Judge' vs 'Chief Judge of the court'), but the title must be explicitly supported. "
              "If the title is missing or empty, mark incorrect."
        ),
    )

    # 3) Dismissing official court/office affiliation
    court_node = evaluator.add_leaf(
        id="dismissing_official_court",
        desc="Provides the court/office the dismissing official serves (i.e., the official’s court affiliation).",
        parent=group_node,
        critical=True,
    )
    court_value = info.official_court or ""
    court_claim = (
        f"The dismissing official serves at or is affiliated with '{court_value}'."
    )
    await evaluator.verify(
        claim=court_claim,
        node=court_node,
        sources=srcs,
        additional_instruction=(
            base_ins
            + " The webpage should clearly state the court or office affiliation of the dismissing official. "
              "If the court/office value is missing or empty, mark incorrect."
        ),
    )

    # 4) Dismissal order date
    date_node = evaluator.add_leaf(
        id="dismissal_order_date",
        desc="Provides the date the dismissal order (dismissing the DOJ misconduct complaint against Judge Ana Reyes) was issued.",
        parent=group_node,
        critical=True,
    )
    date_value = info.dismissal_order_date or ""
    date_claim = (
        f"The dismissal order was issued on '{date_value}'."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=srcs,
        additional_instruction=(
            base_ins
            + " Verify the issuance date of the dismissal order on the cited webpage(s). Accept reasonable date format "
              "variants (e.g., 'Dec. 5, 2025' vs 'December 5, 2025'). If the date is missing or empty, mark incorrect."
        ),
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
    Evaluate an answer for the Ana Reyes dismissal information task.
    """
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

    # Extract dismissal information from the answer
    dismissal_info = await evaluator.extract(
        prompt=prompt_extract_dismissal_info(),
        template_class=DismissalInfo,
        extraction_name="ana_reyes_dismissal_info",
    )

    # Build verification nodes and run checks
    await verify_dismissal_info(
        evaluator=evaluator,
        parent_node=root,
        info=dismissal_info,
    )

    # Return structured result
    return evaluator.get_summary()