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
TASK_ID = "bf2024_store_hours_trip"
TASK_DESCRIPTION = """
You're planning a craft and home improvement shopping trip during Thanksgiving weekend 2024. You want to visit Michaels, Hobby Lobby, and Home Depot stores on Black Friday (November 29, 2024) and need to plan your schedule. Provide the following information:
(1) What time does Michaels open on Black Friday 2024?
(2) What time does Michaels close on Black Friday 2024?
(3) What time does Hobby Lobby open on Black Friday 2024?
(4) What time does Hobby Lobby close on Black Friday 2024?
(5) What time does Home Depot open on Black Friday 2024?
(6) Confirm whether Michaels, Hobby Lobby, and Home Depot are all closed on Thanksgiving Day 2024 (November 28, 2024).
(7) What is the average square footage of Hobby Lobby stores?
Include reference URLs that verify the store hours information.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StoreHoursInfo(BaseModel):
    black_friday_open_time: Optional[str] = None
    black_friday_close_time: Optional[str] = None  # Use None for stores where close time isn't provided
    thanksgiving_closed_2024: Optional[bool] = None  # Whether the answer states the store is closed on Thanksgiving Day 2024
    source_urls: List[str] = Field(default_factory=list)  # URLs cited for store hours/holiday policies


class MichaelsInfo(StoreHoursInfo):
    pass


class HobbyLobbyInfo(StoreHoursInfo):
    average_store_size: Optional[str] = None  # e.g., "55,000 square feet"
    avg_size_source_urls: List[str] = Field(default_factory=list)  # URLs cited for the average size claim


class HomeDepotInfo(StoreHoursInfo):
    pass


class ShoppingPlanExtraction(BaseModel):
    michaels: Optional[MichaelsInfo] = None
    hobby_lobby: Optional[HobbyLobbyInfo] = None
    home_depot: Optional[HomeDepotInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shopping_info() -> str:
    return """
    From the provided answer text, extract exactly the specific details requested for Black Friday (Nov 29, 2024) and Thanksgiving Day (Nov 28, 2024) about Michaels, Hobby Lobby, and Home Depot. Only extract information explicitly present in the answer.

    For each store (Michaels, Hobby Lobby, Home Depot), extract:
    - black_friday_open_time: The opening time on Black Friday 2024 as written in the answer (e.g., "7:00 a.m.", "7 AM", "6 am", "6:00 a.m.", etc.). If not provided, set to null.
    - black_friday_close_time: The closing time on Black Friday 2024 as written in the answer (if present; otherwise null). Note: Home Depot close time may not be provided; return null if not present.
    - thanksgiving_closed_2024: A boolean indicating whether the answer states the store is closed on Thanksgiving Day 2024. If the answer explicitly says it is closed on Thanksgiving, set to true; if it clearly says open, set to false; if not stated, set to null.
    - source_urls: An array of URLs that the answer cites to support store-hours/holiday information for that store (Black Friday hours and/or Thanksgiving closure). Extract actual URLs only. If none are provided, return an empty array.

    Additionally for Hobby Lobby only:
    - average_store_size: The statement about average store size as written in the answer (e.g., "55,000 square feet"). If not provided, set to null.
    - avg_size_source_urls: An array of URLs that support the average store size claim. If none are provided, return an empty array.

    Return a JSON object with the following structure:
    {
      "michaels": {
        "black_friday_open_time": string | null,
        "black_friday_close_time": string | null,
        "thanksgiving_closed_2024": boolean | null,
        "source_urls": string[]
      },
      "hobby_lobby": {
        "black_friday_open_time": string | null,
        "black_friday_close_time": string | null,
        "thanksgiving_closed_2024": boolean | null,
        "source_urls": string[],
        "average_store_size": string | null,
        "avg_size_source_urls": string[]
      },
      "home_depot": {
        "black_friday_open_time": string | null,
        "black_friday_close_time": string | null,
        "thanksgiving_closed_2024": boolean | null,
        "source_urls": string[]
      }
    }

    Notes:
    - Times can be in 12-hour or 24-hour formats; extract them exactly as written in the answer (do not normalize).
    - For URLs: only include valid, explicit URLs found in the answer text. Do not invent or infer URLs. Include markdown link targets if present.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_any_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


def _first_non_empty(*values: Optional[List[str]]) -> List[str]:
    """Return the first list among the inputs that is non-empty; otherwise empty list."""
    for v in values:
        if _has_any_urls(v):
            return v  # type: ignore
    return []


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extraction: ShoppingPlanExtraction,
    logger: logging.Logger,
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """

    # Root node already exists (created by evaluator.initialize). We'll add children under it.

    # 1) Store-hours reference URLs presence checks (critical, parallel)
    store_urls_node = evaluator.add_parallel(
        id="store_hours_reference_urls",
        desc="Provides reference URL(s) that verify the store-hours information claimed (Black Friday hours and/or Thanksgiving closure status).",
        parent=evaluator.root,
        critical=True,
    )

    # Prepare per-store URL presence checks (custom nodes)
    michaels_urls = (extraction.michaels.source_urls if extraction.michaels else [])  # type: ignore
    hobby_lobby_urls = (extraction.hobby_lobby.source_urls if extraction.hobby_lobby else [])  # type: ignore
    home_depot_urls = (extraction.home_depot.source_urls if extraction.home_depot else [])  # type: ignore

    michaels_urls_node = evaluator.add_custom_node(
        result=_has_any_urls(michaels_urls),
        id="michaels_hours_cited",
        desc="Includes at least one reference URL supporting the Michaels Black Friday hours and/or Thanksgiving closure claim.",
        parent=store_urls_node,
        critical=True,
    )

    hobby_lobby_urls_node = evaluator.add_custom_node(
        result=_has_any_urls(hobby_lobby_urls),
        id="hobby_lobby_hours_cited",
        desc="Includes at least one reference URL supporting the Hobby Lobby Black Friday hours and/or Thanksgiving closure claim.",
        parent=store_urls_node,
        critical=True,
    )

    home_depot_urls_node = evaluator.add_custom_node(
        result=_has_any_urls(home_depot_urls),
        id="home_depot_hours_cited",
        desc="Includes at least one reference URL supporting the Home Depot Black Friday opening time and/or Thanksgiving closure claim.",
        parent=store_urls_node,
        critical=True,
    )

    # Collect verification tasks to run in parallel via batch_verify where appropriate
    batch_items: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # Helper to create a leaf, set it failed if missing value, otherwise enqueue verification
    def _add_time_verification_leaf(
        node_id: str,
        desc: str,
        claim_text: Optional[str],
        sources: List[str],
        prerequisites: Optional[List[Any]] = None,
        additional_instruction: Optional[str] = None,
    ):
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=evaluator.root,
            critical=True,
        )
        # If the claim cannot be formed (e.g., missing value), mark leaf as failed directly
        if not claim_text or not claim_text.strip():
            leaf.score = 0.0
            leaf.status = "failed"
            # Record a hint for debugging
            evaluator.add_custom_info(
                {"reason": f"Missing required value for {node_id}; cannot verify."},
                info_type="missing_value",
                info_name=f"{node_id}_missing"
            )
            return

        # Enqueue verification
        batch_items.append((
            claim_text,
            sources if _has_any_urls(sources) else None,  # Route to multi-URL verify only if sources exist
            leaf,
            additional_instruction or "None"
        ))

        # Attach prerequisites if provided (these will be checked inside evaluator.verify)
        if prerequisites:
            # We cannot attach prerequisites directly here; evaluator.verify will be called later by batch
            # To support prerequisites, we will call evaluator.verify individually instead of batch for these.
            pass

    # 2) Michaels Black Friday opening time
    m_open = extraction.michaels.black_friday_open_time if extraction.michaels else None
    _add_time_verification_leaf(
        node_id="michaels_black_friday_opening_time",
        desc="States Michaels opens at the stated time on Black Friday 2024 (Nov 29, 2024).",
        claim_text=(f"Michaels opens at {m_open} on Black Friday 2024 (Nov 29, 2024)." if m_open else None),
        sources=michaels_urls,
        prerequisites=[michaels_urls_node],
        additional_instruction="Allow minor format differences (e.g., 7, 7am, 7:00 AM). Verify the specific Black Friday 2024 opening time."
    )

    # 3) Michaels Black Friday closing time
    m_close = extraction.michaels.black_friday_close_time if extraction.michaels else None
    _add_time_verification_leaf(
        node_id="michaels_black_friday_closing_time",
        desc="States Michaels closes at the stated time on Black Friday 2024 (Nov 29, 2024).",
        claim_text=(f"Michaels closes at {m_close} on Black Friday 2024 (Nov 29, 2024)." if m_close else None),
        sources=michaels_urls,
        prerequisites=[michaels_urls_node],
        additional_instruction="Allow minor format differences (e.g., 10, 10pm, 10:00 PM). Verify the specific Black Friday 2024 closing time."
    )

    # 4) Hobby Lobby Black Friday opening time
    hl_open = extraction.hobby_lobby.black_friday_open_time if extraction.hobby_lobby else None
    _add_time_verification_leaf(
        node_id="hobby_lobby_black_friday_opening_time",
        desc="States Hobby Lobby opens at the stated time on Black Friday 2024 (Nov 29, 2024).",
        claim_text=(f"Hobby Lobby opens at {hl_open} on Black Friday 2024 (Nov 29, 2024)." if hl_open else None),
        sources=hobby_lobby_urls,
        prerequisites=[hobby_lobby_urls_node],
        additional_instruction="Allow minor format differences (e.g., 8, 8am, 8:00 AM). Verify the specific Black Friday 2024 opening time."
    )

    # 5) Hobby Lobby Black Friday closing time
    hl_close = extraction.hobby_lobby.black_friday_close_time if extraction.hobby_lobby else None
    _add_time_verification_leaf(
        node_id="hobby_lobby_black_friday_closing_time",
        desc="States Hobby Lobby closes at the stated time on Black Friday 2024 (Nov 29, 2024).",
        claim_text=(f"Hobby Lobby closes at {hl_close} on Black Friday 2024 (Nov 29, 2024)." if hl_close else None),
        sources=hobby_lobby_urls,
        prerequisites=[hobby_lobby_urls_node],
        additional_instruction="Allow minor format differences (e.g., 9, 9pm, 9:00 PM). Verify the specific Black Friday 2024 closing time."
    )

    # 6) Home Depot Black Friday opening time
    hd_open = extraction.home_depot.black_friday_open_time if extraction.home_depot else None
    _add_time_verification_leaf(
        node_id="home_depot_black_friday_opening_time",
        desc="States Home Depot opens at the stated time on Black Friday 2024 (Nov 29, 2024).",
        claim_text=(f"The Home Depot opens at {hd_open} on Black Friday 2024 (Nov 29, 2024)." if hd_open else None),
        sources=home_depot_urls,
        prerequisites=[home_depot_urls_node],
        additional_instruction="Allow minor format differences (e.g., 6, 6am, 6:00 AM). Verify the specific Black Friday 2024 opening time."
    )

    # 7) Thanksgiving Day 2024 closures (split into per-store leaves under a critical parent)
    tg_parent = evaluator.add_parallel(
        id="thanksgiving_day_2024_closures",
        desc="Confirms Michaels, Hobby Lobby, and Home Depot are all closed on Thanksgiving Day 2024 (Nov 28, 2024).",
        parent=evaluator.root,
        critical=True,
    )

    # Michaels Thanksgiving closure
    tg_m_leaf = evaluator.add_leaf(
        id="thanksgiving_closed_michaels",
        desc="Michaels is closed on Thanksgiving Day 2024 (Nov 28, 2024).",
        parent=tg_parent,
        critical=True,
    )
    # If the answer didn't state closure, still verify against sources; if no sources, it will likely fail.
    claim_tg_m = "Michaels is closed on Thanksgiving Day 2024 (Nov 28, 2024)."
    # Verify this claim with Michaels URLs
    await evaluator.verify(
        claim=claim_tg_m,
        node=tg_m_leaf,
        sources=michaels_urls,
        additional_instruction="The page must clearly indicate that Michaels is closed on Thanksgiving Day (Nov 28, 2024), or generally 'closed on Thanksgiving'."
    )

    # Hobby Lobby Thanksgiving closure
    tg_hl_leaf = evaluator.add_leaf(
        id="thanksgiving_closed_hobby_lobby",
        desc="Hobby Lobby is closed on Thanksgiving Day 2024 (Nov 28, 2024).",
        parent=tg_parent,
        critical=True,
    )
    claim_tg_hl = "Hobby Lobby is closed on Thanksgiving Day 2024 (Nov 28, 2024)."
    await evaluator.verify(
        claim=claim_tg_hl,
        node=tg_hl_leaf,
        sources=hobby_lobby_urls,
        additional_instruction="The page must clearly indicate that Hobby Lobby is closed on Thanksgiving Day (Nov 28, 2024), or generally 'closed on Thanksgiving'."
    )

    # Home Depot Thanksgiving closure
    tg_hd_leaf = evaluator.add_leaf(
        id="thanksgiving_closed_home_depot",
        desc="Home Depot is closed on Thanksgiving Day 2024 (Nov 28, 2024).",
        parent=tg_parent,
        critical=True,
    )
    claim_tg_hd = "The Home Depot is closed on Thanksgiving Day 2024 (Nov 28, 2024)."
    await evaluator.verify(
        claim=claim_tg_hd,
        node=tg_hd_leaf,
        sources=home_depot_urls,
        additional_instruction="The page must clearly indicate that Home Depot is closed on Thanksgiving Day (Nov 28, 2024), or generally 'closed on Thanksgiving'."
    )

    # 8) Hobby Lobby average store size
    hl_avg_size = extraction.hobby_lobby.average_store_size if extraction.hobby_lobby else None
    size_sources = _first_non_empty(
        extraction.hobby_lobby.avg_size_source_urls if extraction.hobby_lobby else [],  # type: ignore
        hobby_lobby_urls
    )
    size_leaf = evaluator.add_leaf(
        id="hobby_lobby_average_store_size",
        desc="States Hobby Lobby stores average the stated square footage in size.",
        parent=evaluator.root,
        critical=True,
    )
    if not hl_avg_size or not hl_avg_size.strip():
        size_leaf.score = 0.0
        size_leaf.status = "failed"
        evaluator.add_custom_info(
            {"reason": "Missing average store size statement for Hobby Lobby; cannot verify."},
            info_type="missing_value",
            info_name="hobby_lobby_average_store_size_missing"
        )
    else:
        claim_size = f"Hobby Lobby stores average {hl_avg_size} in size."
        await evaluator.verify(
            claim=claim_size,
            node=size_leaf,
            sources=size_sources if _has_any_urls(size_sources) else None,
            additional_instruction="Accept equivalent phrasings like 'average store size' or 'typical store footprint'."
        )

    # Run the queued time verifications (those with proper values)
    # Note: Some leaves may depend on presence of URLs; however, we placed the URL presence checks upfront.
    if batch_items:
        await evaluator.batch_verify(batch_items)


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
    Evaluate an answer for the Black Friday 2024 store hours and Thanksgiving closure task.
    """
    # Initialize evaluator (root is parallel by default in our design; criticality of children will enforce gating)
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

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_shopping_info(),
        template_class=ShoppingPlanExtraction,
        extraction_name="extracted_store_info",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extraction, logger)

    # Return structured result
    return evaluator.get_summary()