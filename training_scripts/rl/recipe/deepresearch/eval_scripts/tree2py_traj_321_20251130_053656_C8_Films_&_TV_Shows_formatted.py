import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "releases_2024_2025_platforms"
TASK_DESCRIPTION = (
    "Identify four television shows or films that premiered or were theatrically released between June 1, 2024 "
    "and November 30, 2025 (inclusive), where each work satisfies ALL of the following requirements:\n\n"
    "1. Has a documented and verifiable premiere or theatrical release date within the specified time window\n"
    "2. Features at least one cast member in a specific credited role that can be verified\n"
    "3. Was released through a specific platform or format (such as theatrical release, streaming platform premiere, or broadcast television premiere)\n"
    "4. The collection of four works must include at least one theatrical film, at least one streaming series, and at least one broadcast television show\n\n"
    "For each identified work, provide:\n"
    "- The complete title of the work\n"
    "- The exact premiere or release date (month, day, and year)\n"
    "- The name of at least one cast member and their specific credited role in that work\n"
    "- The platform or format through which it was released (e.g., theatrical release, Netflix, ABC, etc.)\n"
    "- A reference URL that verifies this information"
)

WINDOW_START = "June 1, 2024"
WINDOW_END = "November 30, 2025"

STREAMING_PLATFORMS = {
    "netflix", "hulu", "max", "hbo max", "disney+", "disney plus", "prime video", "amazon prime video",
    "apple tv+", "apple tv plus", "paramount+", "peacock", "starz", "showtime", "crunchyroll"
}
BROADCAST_NETWORKS = {
    "abc", "cbs", "nbc", "fox", "the cw", "cw", "pbs", "itv", "bbc one", "bbc two", "channel 4", "tvn", "tvp"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CastCredit(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None


class WorkItem(BaseModel):
    title: Optional[str] = None
    date: Optional[str] = None  # Month Day, Year string preferred
    date_kind: Optional[str] = None  # e.g., "theatrical release", "series premiere", "broadcast premiere", etc.
    cast: Optional[CastCredit] = None
    platform_label: Optional[str] = None  # e.g., "Netflix", "ABC", "Theatrical release"
    platform_type: Optional[str] = None  # one of: "theatrical_film", "streaming_series", "broadcast_tv", "other"
    urls: List[str] = Field(default_factory=list)


class WorksExtraction(BaseModel):
    works: List[WorkItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_works() -> str:
    return """
    From the provided answer, extract all mentioned works (films or television shows) that include details about release/premiere, cast, and platform. Return a JSON object with a field "works" which is a list of objects. For each work, extract:

    - title: The complete title of the work as stated in the answer.
    - date: The exact date (month day, year) of the premiere or theatrical release as stated (e.g., "September 12, 2024"). If multiple dates are mentioned, choose the one relevant to the described release/premiere event for that work.
    - date_kind: The type of the date (e.g., "theatrical release", "series premiere", "broadcast premiere", "streaming premiere", "season premiere"). Use the phrasing that best matches the answer.
    - cast: An object with:
        - name: The name of one cast member (or participant/host/judge for non-scripted shows).
        - role: The specific credited role for that person in this work (e.g., "as Jane Doe", "host", "contestant", "voice of X").
    - platform_label: The platform or format label (e.g., "theatrical release", "Netflix", "ABC", "BBC One", "Disney+", "CBS").
    - platform_type: A classification inferred from the answer; choose exactly one of:
        - "theatrical_film" (released in cinemas/movie theaters)
        - "streaming_series" (premiered on a streaming platform like Netflix, Hulu, Disney+, Prime Video, Apple TV+, Max, Paramount+, Peacock, etc.)
        - "broadcast_tv" (premiered on a broadcast TV network such as ABC, CBS, NBC, FOX, The CW, PBS, BBC One, etc.)
        - "other" (if none of the above clearly apply)
    - urls: An array of reference URLs explicitly provided in the answer that support the information for this work. Extract actual URLs (expand markdown links). If no URLs are given in the answer, return an empty array.

    Important:
    - Only extract what is explicitly present in the answer. Do not invent fields. If any field is missing from the answer, set it to null (or [] for urls).
    - Prefer the premiere or theatrical release date for the initial public release/premiere event, as described by the answer. If the answer specifically refers to theatrical release (cinemas), choose that date.
    - Keep exactly one cast member with a specific role; if multiple are present, choose one.
    - Normalize platform_type as specified.
    """


# --------------------------------------------------------------------------- #
# Selection helpers                                                           #
# --------------------------------------------------------------------------- #
def _matches_type(item: WorkItem, desired: str) -> bool:
    if not item:
        return False
    # Direct classification
    if item.platform_type and item.platform_type.strip().lower() == desired:
        return True
    # Heuristics if classification missing/ambiguous
    lbl = (item.platform_label or "").strip().lower()
    if desired == "streaming_series":
        return lbl in STREAMING_PLATFORMS or any(s in lbl for s in STREAMING_PLATFORMS)
    if desired == "broadcast_tv":
        return lbl in BROADCAST_NETWORKS or any(s in lbl for s in BROADCAST_NETWORKS)
    if desired == "theatrical_film":
        return ("theatrical" in lbl) or ("cinema" in lbl) or ("in theaters" in lbl) or ("in theatres" in lbl)
    return False


def _pick_item(works: List[WorkItem], desired_type: str, used: set) -> Optional[int]:
    # First pass: direct match
    for idx, w in enumerate(works):
        if idx in used:
            continue
        if _matches_type(w, desired_type):
            return idx
    # No direct match; return None
    return None


def _pick_any_remaining(works: List[WorkItem], used: set) -> Optional[int]:
    for idx, w in enumerate(works):
        if idx in used:
            continue
        # Prefer items with at least a title and a url to increase chance of verifiability
        if (w.title and w.title.strip()) and (w.urls and any((u or "").strip() for u in w.urls)):
            return idx
    # fallback: first completely unused
    for idx, _ in enumerate(works):
        if idx not in used:
            return idx
    return None


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def _verify_theatrical_film(
    evaluator: Evaluator,
    parent_node,
    item: WorkItem,
) -> None:
    # Parent node: theatrical film (already added by caller)
    # 1) Reference URL provided
    ref_ok = bool(item and item.urls and any((u or "").strip() for u in item.urls))
    ref_node = evaluator.add_custom_node(
        result=ref_ok,
        id="theatrical_film_reference",
        desc="Reference URL provided that verifies the theatrical film information",
        parent=parent_node,
        critical=True,
    )

    # 2) Title provided
    title_ok = bool(item and item.title and item.title.strip())
    title_node = evaluator.add_custom_node(
        result=title_ok,
        id="theatrical_film_title",
        desc="The complete title of the theatrical film is provided",
        parent=parent_node,
        critical=True,
    )

    # Prepare sources and title for claims
    sources = item.urls if (item and item.urls) else None
    claimed_title = (item.title or "this film").strip() if item else "this film"

    # 3) Release date within range (verify with sources)
    date_leaf = evaluator.add_leaf(
        id="theatrical_film_release_date",
        desc="The theatrical film has a documented release date between June 1, 2024 and November 30, 2025",
        parent=parent_node,
        critical=True,
    )
    date_text = (item.date or "").strip() if item else ""
    date_kind = (item.date_kind or "theatrical release").strip() if item else "theatrical release"
    date_claim = (
        f"The theatrical release date of '{claimed_title}' is {date_text}, and this date falls between "
        f"{WINDOW_START} and {WINDOW_END} (inclusive)."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=sources,
        additional_instruction=(
            "Use the provided source(s) to confirm the film's theatrical release date. "
            "If multiple regions/dates are shown, it's acceptable if at least one theatrical release date "
            f"occurs within {WINDOW_START} and {WINDOW_END}."
        ),
        extra_prerequisites=[ref_node, title_node],
    )

    # 4) Cast member with specific role
    cast_leaf = evaluator.add_leaf(
        id="theatrical_film_cast_member",
        desc="At least one cast member is identified with their specific credited role in the theatrical film",
        parent=parent_node,
        critical=True,
    )
    cast_name = (item.cast.name if item and item.cast and item.cast.name else "").strip()
    cast_role = (item.cast.role if item and item.cast and item.cast.role else "").strip()
    cast_claim = f"'{cast_name}' is credited as '{cast_role}' in the film '{claimed_title}'."
    await evaluator.verify(
        claim=cast_claim,
        node=cast_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the cast/credits section for the named individual and the specific role/character. "
            "Minor formatting differences are acceptable (e.g., case, punctuation)."
        ),
        extra_prerequisites=[ref_node, title_node],
    )

    # 5) Platform/format: verified as theatrical film release
    platform_leaf = evaluator.add_leaf(
        id="theatrical_film_platform",
        desc="The work is verified as a theatrical film release",
        parent=parent_node,
        critical=True,
    )
    platform_claim = f"'{claimed_title}' was released theatrically (in cinemas/movie theaters)."
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        sources=sources,
        additional_instruction="Confirm from the source(s) that this work had a theatrical release.",
        extra_prerequisites=[ref_node, title_node],
    )


async def _verify_streaming_series(
    evaluator: Evaluator,
    parent_node,
    item: WorkItem,
) -> None:
    # 1) Reference URL provided
    ref_ok = bool(item and item.urls and any((u or "").strip() for u in item.urls))
    ref_node = evaluator.add_custom_node(
        result=ref_ok,
        id="streaming_series_reference",
        desc="Reference URL provided that verifies the streaming series information",
        parent=parent_node,
        critical=True,
    )

    # 2) Title provided
    title_ok = bool(item and item.title and item.title.strip())
    title_node = evaluator.add_custom_node(
        result=title_ok,
        id="streaming_series_title",
        desc="The complete title of the streaming series is provided",
        parent=parent_node,
        critical=True,
    )

    sources = item.urls if (item and item.urls) else None
    claimed_title = (item.title or "this series").strip() if item else "this series"

    # 3) Premiere date in range
    date_leaf = evaluator.add_leaf(
        id="streaming_series_premiere_date",
        desc="The streaming series has a documented premiere date between June 1, 2024 and November 30, 2025",
        parent=parent_node,
        critical=True,
    )
    date_text = (item.date or "").strip() if item else ""
    date_claim = (
        f"The series premiere date of '{claimed_title}' is {date_text}, and this date falls between "
        f"{WINDOW_START} and {WINDOW_END} (inclusive)."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the date shown is the series premiere (first public release on the streaming platform). "
            f"Accept if it falls within {WINDOW_START} and {WINDOW_END}."
        ),
        extra_prerequisites=[ref_node, title_node],
    )

    # 4) Cast member with specific role
    cast_leaf = evaluator.add_leaf(
        id="streaming_series_cast_member",
        desc="At least one cast member is identified with their specific credited role in the streaming series",
        parent=parent_node,
        critical=True,
    )
    cast_name = (item.cast.name if item and item.cast and item.cast.name else "").strip()
    cast_role = (item.cast.role if item and item.cast and item.cast.role else "").strip()
    cast_claim = f"'{cast_name}' is credited as '{cast_role}' in the series '{claimed_title}'."
    await evaluator.verify(
        claim=cast_claim,
        node=cast_leaf,
        sources=sources,
        additional_instruction="Verify the credits for the named cast member and their specific role/character.",
        extra_prerequisites=[ref_node, title_node],
    )

    # 5) Platform: verified as streaming platform
    platform_leaf = evaluator.add_leaf(
        id="streaming_series_platform",
        desc="The work is verified as premiering on a streaming platform",
        parent=parent_node,
        critical=True,
    )
    platform_label = (item.platform_label or "a streaming platform").strip() if item else "a streaming platform"
    platform_claim = f"'{claimed_title}' premiered on {platform_label}, which is a streaming platform."
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the platform is a streaming service (e.g., Netflix, Hulu, Disney+, Prime Video, Apple TV+, Max, Paramount+, Peacock)."
        ),
        extra_prerequisites=[ref_node, title_node],
    )


async def _verify_broadcast_show(
    evaluator: Evaluator,
    parent_node,
    item: WorkItem,
) -> None:
    # 1) Reference URL provided
    ref_ok = bool(item and item.urls and any((u or "").strip() for u in item.urls))
    ref_node = evaluator.add_custom_node(
        result=ref_ok,
        id="broadcast_show_reference",
        desc="Reference URL provided that verifies the broadcast television show information",
        parent=parent_node,
        critical=True,
    )

    # 2) Title provided
    title_ok = bool(item and item.title and item.title.strip())
    title_node = evaluator.add_custom_node(
        result=title_ok,
        id="broadcast_show_title",
        desc="The complete title of the broadcast television show is provided",
        parent=parent_node,
        critical=True,
    )

    sources = item.urls if (item and item.urls) else None
    claimed_title = (item.title or "this broadcast show").strip() if item else "this broadcast show"

    # 3) Premiere date in range
    date_leaf = evaluator.add_leaf(
        id="broadcast_show_premiere_date",
        desc="The broadcast television show has a documented premiere date between June 1, 2024 and November 30, 2025",
        parent=parent_node,
        critical=True,
    )
    date_text = (item.date or "").strip() if item else ""
    date_claim = (
        f"The broadcast television premiere date of '{claimed_title}' is {date_text}, and this date falls between "
        f"{WINDOW_START} and {WINDOW_END} (inclusive)."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the cited date is the broadcast TV premiere on a broadcast network and that it lies within the window."
        ),
        extra_prerequisites=[ref_node, title_node],
    )

    # 4) Cast member or participant with specific role
    cast_leaf = evaluator.add_leaf(
        id="broadcast_show_cast_member",
        desc="At least one cast member or contestant is identified with their specific participation or role in the broadcast show",
        parent=parent_node,
        critical=True,
    )
    cast_name = (item.cast.name if item and item.cast and item.cast.name else "").strip()
    cast_role = (item.cast.role if item and item.cast and item.cast.role else "").strip()
    cast_claim = f"'{cast_name}' is credited as '{cast_role}' in the broadcast show '{claimed_title}'."
    await evaluator.verify(
        claim=cast_claim,
        node=cast_leaf,
        sources=sources,
        additional_instruction="Verify from the source(s) that the named person is credited with the specified role/participation.",
        extra_prerequisites=[ref_node, title_node],
    )

    # 5) Platform: verified as broadcast television network
    platform_leaf = evaluator.add_leaf(
        id="broadcast_show_platform",
        desc="The work is verified as premiering on a broadcast television network",
        parent=parent_node,
        critical=True,
    )
    platform_label = (item.platform_label or "a broadcast network").strip() if item else "a broadcast network"
    platform_claim = f"'{claimed_title}' premiered on {platform_label}, which is a broadcast television network."
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm from the source(s) that the platform is a broadcast TV network (e.g., ABC, CBS, NBC, FOX, The CW, PBS)."
        ),
        extra_prerequisites=[ref_node, title_node],
    )


async def _verify_fourth_work(
    evaluator: Evaluator,
    parent_node,
    item: WorkItem,
) -> None:
    # 1) Reference URL provided
    ref_ok = bool(item and item.urls and any((u or "").strip() for u in item.urls))
    ref_node = evaluator.add_custom_node(
        result=ref_ok,
        id="fourth_work_reference",
        desc="Reference URL provided that verifies the fourth work's information",
        parent=parent_node,
        critical=True,
    )

    # 2) Title provided
    title_ok = bool(item and item.title and item.title.strip())
    title_node = evaluator.add_custom_node(
        result=title_ok,
        id="fourth_work_title",
        desc="The complete title of the fourth work is provided",
        parent=parent_node,
        critical=True,
    )

    sources = item.urls if (item and item.urls) else None
    claimed_title = (item.title or "this work").strip() if item else "this work"

    # 3) Premiere/release date in range
    date_leaf = evaluator.add_leaf(
        id="fourth_work_release_date",
        desc="The fourth work has a documented premiere or release date between June 1, 2024 and November 30, 2025",
        parent=parent_node,
        critical=True,
    )
    date_text = (item.date or "").strip() if item else ""
    date_kind = (item.date_kind or "premiere or release").strip() if item else "premiere or release"
    date_claim = (
        f"The {date_kind} date of '{claimed_title}' is {date_text}, and this date falls between "
        f"{WINDOW_START} and {WINDOW_END} (inclusive)."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the exact premiere/release date from the source(s) and verify it lies within the specified window."
        ),
        extra_prerequisites=[ref_node, title_node],
    )

    # 4) Cast member with role
    cast_leaf = evaluator.add_leaf(
        id="fourth_work_cast_member",
        desc="At least one cast member or participant is identified with their specific credited role or participation in the fourth work",
        parent=parent_node,
        critical=True,
    )
    cast_name = (item.cast.name if item and item.cast and item.cast.name else "").strip()
    cast_role = (item.cast.role if item and item.cast and item.cast.role else "").strip()
    cast_claim = f"'{cast_name}' is credited as '{cast_role}' in '{claimed_title}'."
    await evaluator.verify(
        claim=cast_claim,
        node=cast_leaf,
        sources=sources,
        additional_instruction="Verify the named person and their specific role from the source(s).",
        extra_prerequisites=[ref_node, title_node],
    )

    # 5) Platform/format documented
    platform_leaf = evaluator.add_leaf(
        id="fourth_work_platform",
        desc="The platform or format through which the fourth work was released is clearly documented",
        parent=parent_node,
        critical=True,
    )
    platform_label = (item.platform_label or "").strip() if item else ""
    if platform_label:
        platform_claim = f"The platform or format for '{claimed_title}' is '{platform_label}', and it is clearly documented by the source."
        add_ins = "Confirm that the platform/format label is present and clearly indicated by the source(s)."
    else:
        # Fall back to a generic but stricter claim (likely to fail if platform not documented)
        platform_claim = f"The platform or format for '{claimed_title}' is clearly documented by the source."
        add_ins = "Confirm that the source explicitly names the platform/format (e.g., a streaming service, broadcast network, or theatrical release)."
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        sources=sources,
        additional_instruction=add_ins,
        extra_prerequisites=[ref_node, title_node],
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
    # Initialize evaluator (root: parallel aggregation)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Evaluation of four films or TV shows meeting specified criteria for 2024-2025 releases",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract all works from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_works(),
        template_class=WorksExtraction,
        extraction_name="works_extraction",
    )

    works = extracted.works or []
    used_indices: set = set()

    # Select one of each required type
    idx_film = _pick_item(works, "theatrical_film", used_indices)
    if idx_film is not None:
        used_indices.add(idx_film)
    idx_streaming = _pick_item(works, "streaming_series", used_indices)
    if idx_streaming is not None:
        used_indices.add(idx_streaming)
    idx_broadcast = _pick_item(works, "broadcast_tv", used_indices)
    if idx_broadcast is not None:
        used_indices.add(idx_broadcast)
    # Fourth work: any remaining item
    idx_fourth = _pick_any_remaining(works, used_indices)
    if idx_fourth is not None:
        used_indices.add(idx_fourth)

    # Prepare chosen items (fallback to empty if missing)
    film_item = works[idx_film] if idx_film is not None else WorkItem()
    streaming_item = works[idx_streaming] if idx_streaming is not None else WorkItem()
    broadcast_item = works[idx_broadcast] if idx_broadcast is not None else WorkItem()
    fourth_item = works[idx_fourth] if idx_fourth is not None else WorkItem()

    # Record selection as custom debug info
    evaluator.add_custom_info(
        info={
            "selected_items": {
                "theatrical_film": film_item.dict(),
                "streaming_series": streaming_item.dict(),
                "broadcast_television_show": broadcast_item.dict(),
                "fourth_work": fourth_item.dict(),
            }
        },
        info_type="selection_debug",
        info_name="selected_works"
    )

    # Build verification tree according to rubric
    # Theatrical film node
    theatrical_node = evaluator.add_parallel(
        id="theatrical_film",
        desc="Identification and verification of one theatrical film released between June 2024 and November 2025",
        parent=root,
        critical=False,
    )
    await _verify_theatrical_film(evaluator, theatrical_node, film_item)

    # Streaming series node
    streaming_node = evaluator.add_parallel(
        id="streaming_series",
        desc="Identification and verification of one streaming series that premiered between June 2024 and November 2025",
        parent=root,
        critical=False,
    )
    await _verify_streaming_series(evaluator, streaming_node, streaming_item)

    # Broadcast television show node
    broadcast_node = evaluator.add_parallel(
        id="broadcast_television_show",
        desc="Identification and verification of one broadcast television show that premiered between June 2024 and November 2025",
        parent=root,
        critical=False,
    )
    await _verify_broadcast_show(evaluator, broadcast_node, broadcast_item)

    # Fourth work node (any format)
    fourth_node = evaluator.add_parallel(
        id="fourth_work",
        desc="Identification and verification of a fourth film or TV show (any format) released between June 2024 and November 2025",
        parent=root,
        critical=False,
    )
    await _verify_fourth_work(evaluator, fourth_node, fourth_item)

    # Return structured evaluation summary
    return evaluator.get_summary()