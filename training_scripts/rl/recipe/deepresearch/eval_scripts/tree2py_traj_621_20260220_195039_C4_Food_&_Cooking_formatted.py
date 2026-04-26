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
TASK_ID = "grocery_holiday_hours_2025_2026"
TASK_DESCRIPTION = (
    "Identify a national grocery store chain in the United States that meets all of the following holiday operating "
    "requirements for 2025-2026: (1) The chain must be open on Thanksgiving Day (November 27, 2025), "
    "(2) The chain must operate with reduced hours on Thanksgiving Day, closing before 6:00 PM local time, "
    "(3) The pharmacy departments within the chain's stores must be closed on Thanksgiving Day, "
    "(4) The chain must be open on New Year's Day (January 1, 2026), "
    "(5) The chain must operate beyond 5:00 PM on New Year's Day (January 1, 2026). "
    "Provide the name of the grocery store chain and an official reference URL that confirms the holiday hours information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StoreHolidayExtraction(BaseModel):
    """
    Extracted information from the agent's answer for the holiday hours task.
    """
    chain_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_store_info() -> str:
    return (
        "From the answer, extract the following fields:\n"
        "1. chain_name: The name of the national grocery store chain in the United States that the answer claims "
        "   meets the specified holiday operating requirements.\n"
        "2. reference_urls: All official URL(s) provided in the answer that are intended to confirm the chain's "
        "   holiday hours information (e.g., the chain's official website pages such as holiday hours pages, "
        "   customer service/FAQ on holiday hours, official announcements/press releases about holiday schedules). "
        "   Only include URLs explicitly present in the answer; do not infer or invent URLs. Prefer official domains "
        "   belonging to the chain (e.g., chainname.com). Exclude third-party aggregators unless explicitly cited.\n\n"
        "If the answer mentions multiple URLs, include them all in the list. If a field is missing, set it to null "
        "for chain_name and return an empty list for reference_urls."
    )


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_requirements(
    evaluator: Evaluator,
    parent_node,
    extracted: StoreHolidayExtraction,
) -> None:
    """
    Build and verify the holiday hours requirement tree under a critical parallel parent node.
    """
    chain_name = (extracted.chain_name or "").strip()
    urls: List[str] = [u for u in (extracted.reference_urls or []) if isinstance(u, str) and u.strip()]

    # Create a critical parent node to mirror rubric root critical behavior
    main = evaluator.add_parallel(
        id="holiday_requirements",
        desc="Identify a national grocery store chain that meets all specified holiday operating requirements",
        parent=parent_node,
        critical=True,
    )

    # Existence checks (critical preconditions)
    evaluator.add_custom_node(
        result=bool(chain_name),
        id="chain_name_provided",
        desc="A grocery store chain name is provided in the answer",
        parent=main,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(urls),
        id="reference_url_provided",
        desc="At least one official reference URL is provided in the answer",
        parent=main,
        critical=True,
    )

    # Reference URL officialness and relevance (critical)
    ref_leaf = evaluator.add_leaf(
        id="reference_url_official",
        desc="Provided reference URL is official and presents holiday hours information for the chain",
        parent=main,
        critical=True,
    )
    ref_claim = (
        f"The provided URL is an official page from {chain_name} (i.e., owned by the chain) and "
        f"explicitly presents holiday hours or holiday schedule information relevant to late 2025 / early 2026."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=urls,
        additional_instruction=(
            "Verify the URL belongs to the chain (official domain / branding) and the page contains explicit "
            "holiday hours information (e.g., Thanksgiving/New Year's or a holiday schedule page). "
            "If multiple URLs are provided, any one being official and showing holiday hours suffices."
        ),
    )

    # Thanksgiving Day: open (critical)
    tg_open_leaf = evaluator.add_leaf(
        id="thanksgiving_open",
        desc="The grocery store chain is open on Thanksgiving Day (November 27, 2025)",
        parent=main,
        critical=True,
    )
    tg_open_claim = f"{chain_name} stores are open on Thanksgiving Day (November 27, 2025)."
    await evaluator.verify(
        claim=tg_open_claim,
        node=tg_open_leaf,
        sources=urls,
        additional_instruction=(
            "Check the holiday hours/schedule for Thanksgiving Day 2025. "
            "The page should state that stores are open that day (even if with modified/reduced hours)."
        ),
    )

    # Thanksgiving Day: reduced hours AND closing before 6 PM (split into two critical leaves)
    tg_hours_group = evaluator.add_parallel(
        id="thanksgiving_hours",
        desc="Thanksgiving Day operations: reduced hours and closing before 6:00 PM local time",
        parent=main,
        critical=True,
    )

    tg_reduced_leaf = evaluator.add_leaf(
        id="thanksgiving_reduced_hours",
        desc="On Thanksgiving Day 2025, the chain operates with reduced/shortened hours",
        parent=tg_hours_group,
        critical=True,
    )
    tg_reduced_claim = (
        f"On Thanksgiving Day (November 27, 2025), {chain_name} operates with reduced or shortened hours "
        f"(e.g., opens later and/or closes earlier than typical)."
    )
    await evaluator.verify(
        claim=tg_reduced_claim,
        node=tg_reduced_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm that the holiday page indicates modified/reduced/special hours for Thanksgiving Day. "
            "Explicit phrases like 'reduced hours', 'special holiday hours', or a notably shorter span compared to normal "
            "qualify as reduced hours."
        ),
    )

    tg_before_6_leaf = evaluator.add_leaf(
        id="thanksgiving_closes_before_6pm",
        desc="On Thanksgiving Day 2025, stores close before 6:00 PM local time",
        parent=tg_hours_group,
        critical=True,
    )
    tg_before_6_claim = (
        f"On Thanksgiving Day (November 27, 2025), {chain_name} stores close before 6:00 PM local time."
    )
    await evaluator.verify(
        claim=tg_before_6_claim,
        node=tg_before_6_leaf,
        sources=urls,
        additional_instruction=(
            "Check the listed Thanksgiving closing time. Any closing time strictly earlier than 6:00 PM "
            "(e.g., 3 PM, 4 PM, 5 PM) satisfies this requirement. Consider local times."
        ),
    )

    # Thanksgiving Day: pharmacy closed (critical)
    tg_pharm_leaf = evaluator.add_leaf(
        id="thanksgiving_pharmacy",
        desc="The pharmacy departments within the grocery store chain are closed on Thanksgiving Day",
        parent=main,
        critical=True,
    )
    tg_pharm_claim = f"On Thanksgiving Day (November 27, 2025), {chain_name} in-store pharmacy departments are closed."
    await evaluator.verify(
        claim=tg_pharm_claim,
        node=tg_pharm_leaf,
        sources=urls,
        additional_instruction=(
            "Verify the holiday policy for in-store pharmacies on Thanksgiving. "
            "Explicit statements like 'pharmacies closed' or 'pharmacy not open' qualify."
        ),
    )

    # New Year's Day: open (critical)
    ny_open_leaf = evaluator.add_leaf(
        id="new_years_open",
        desc="The grocery store chain is open on New Year's Day (January 1, 2026)",
        parent=main,
        critical=True,
    )
    ny_open_claim = f"{chain_name} stores are open on New Year's Day (January 1, 2026)."
    await evaluator.verify(
        claim=ny_open_claim,
        node=ny_open_leaf,
        sources=urls,
        additional_instruction=(
            "Check the holiday hours/schedule for New Year's Day 2026. "
            "The page should indicate stores are open on January 1, 2026."
        ),
    )

    # New Year's Day: operates beyond 5 PM (critical)
    ny_after_5_leaf = evaluator.add_leaf(
        id="new_years_extended",
        desc="The grocery store chain operates beyond 5:00 PM on New Year's Day",
        parent=main,
        critical=True,
    )
    ny_after_5_claim = (
        f"On New Year's Day (January 1, 2026), {chain_name} stores operate beyond 5:00 PM local time "
        f"(i.e., they remain open after 5:00 PM)."
    )
    await evaluator.verify(
        claim=ny_after_5_claim,
        node=ny_after_5_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm that the closing time on New Year's Day is strictly later than 5:00 PM "
            "(e.g., 6 PM, 7 PM, 8 PM). Consider local times."
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an agent's answer for the grocery holiday hours requirements task (2025-2026).
    """
    # Initialize evaluator with a parallel root
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

    # Extract store name and reference URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_store_info(),
        template_class=StoreHolidayExtraction,
        extraction_name="store_holiday_extraction",
    )

    # Build verification tree and run checks
    await verify_requirements(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()