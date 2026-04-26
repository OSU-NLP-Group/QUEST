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
TASK_ID = "np_winter_accessible_lodges_2025_2026"
TASK_DESCRIPTION = (
    "Identify three U.S. National Park lodges that meet ALL of the following criteria for winter 2025-2026: "
    "(1) Winter Operation - The lodge must be open and accepting overnight guests during the period from December 2025 "
    "through February 2026; (2) ADA-Compliant Accessible Rooms - The lodge must offer ADA-compliant accessible guest rooms; "
    "(3) Roll-in Shower - Accessible rooms must include roll-in shower facilities; "
    "(4) Doorway Width - Accessible rooms must have doorways with minimum 32-inch clear width; "
    "(5) Visual Alarms - Accessible rooms must include visual fire alarm systems for hearing-impaired guests; "
    "(6) Grab Bars - Accessible bathrooms must include grab bars in toilet areas; "
    "(7) In-Park Location - The lodge must be located within a U.S. National Park boundary. "
    "For each lodge, provide: lodge name and national park location, winter operating dates, confirmation of all required "
    "accessibility features with specific details, winter vehicle access method, booking contact information for accessible room "
    "reservations, minimum stay requirements if applicable, and reference URLs documenting winter operations and accessibility features."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LodgeAccessibility(BaseModel):
    ada_accessible_rooms: Optional[str] = None
    roll_in_shower: Optional[str] = None
    doorway_clear_width: Optional[str] = None
    visual_alarms: Optional[str] = None
    grab_bars: Optional[str] = None


class LodgeSources(BaseModel):
    winter_ops_urls: List[str] = Field(default_factory=list)
    accessibility_urls: List[str] = Field(default_factory=list)
    boundary_urls: List[str] = Field(default_factory=list)
    access_method_urls: List[str] = Field(default_factory=list)
    booking_urls: List[str] = Field(default_factory=list)


class LodgeItem(BaseModel):
    lodge_name: Optional[str] = None
    national_park: Optional[str] = None
    winter_operating_dates: Optional[str] = None
    winter_access_method: Optional[str] = None
    booking_contact: Optional[str] = None
    minimum_stay: Optional[str] = None  # Accept "not specified"/"not applicable" or null
    accessibility: Optional[LodgeAccessibility] = None
    sources: LodgeSources = Field(default_factory=LodgeSources)


class LodgesExtraction(BaseModel):
    lodges: List[LodgeItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_lodges() -> str:
    return """
    Extract up to the first three distinct lodges explicitly provided in the answer that the answer claims are located within a U.S. National Park and satisfy the winter 2025–2026 and accessibility requirements. For each lodge, return the following fields from the answer exactly as stated (do not invent or infer):
    - lodge_name: The lodge/hotel name
    - national_park: The U.S. National Park name where the lodge is located
    - winter_operating_dates: The text or dates the answer states for winter operations (e.g., "Dec 15, 2025 – Feb 28, 2026", or "open in Dec 2025 through Feb 2026")
    - winter_access_method: The answer's stated winter access method (e.g., "personal vehicle", "snowcoach", "oversnow transportation", "shuttle"), if provided
    - booking_contact: The answer's stated booking contact or method specifically for accessible room reservations (e.g., phone, email, booking page URL, ADA booking instructions); if not provided, set to null
    - minimum_stay: Minimum stay requirement if the answer explicitly provides one, otherwise:
        • If the answer explicitly says "not specified" or "not applicable", return exactly that phrase.
        • If the answer does not mention minimum stay at all, return null.
    - accessibility: Object with the following optional text fields extracted from the answer (set null if not stated):
        • ada_accessible_rooms: A statement confirming ADA-compliant accessible rooms exist.
        • roll_in_shower: A statement confirming roll-in shower exists in accessible rooms.
        • doorway_clear_width: A statement confirming doorways meet at least 32-inch clear width (e.g., "32-inch minimum" or "34-inch doorways").
        • visual_alarms: A statement confirming visual fire alarms (e.g., strobe) are present in accessible rooms.
        • grab_bars: A statement confirming toilet-area grab bars in accessible bathrooms.
    - sources: Object with arrays of explicit URLs mentioned in the answer:
        • winter_ops_urls: URLs that document winter operation or operating dates/schedule.
        • accessibility_urls: URLs that document accessibility features.
        • boundary_urls: URLs that document the lodge is located within the National Park boundary (e.g., NPS site or concessioner page stating "in-park").
        • access_method_urls: URLs that document the winter access method.
        • booking_urls: URLs that document booking/contact for accessible rooms.

    Rules:
    - Only extract URLs explicitly present in the answer (plain or markdown links). If none for a category, return an empty array.
    - Do not infer URLs or data. Use null for missing fields where appropriate.
    - Prefer full official URLs when multiple are present; however, include all URLs mentioned in the answer for each category.
    - Return a JSON object: { "lodges": [ ... up to first 3 ] }.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _merge_urls(*url_lists: List[str]) -> List[str]:
    unique = []
    seen = set()
    for lst in url_lists:
        for u in lst:
            if u and u not in seen:
                unique.append(u)
                seen.add(u)
    return unique


def _sources_for_boundary(lodge: LodgeItem) -> List[str]:
    return _merge_urls(lodge.sources.boundary_urls, lodge.sources.winter_ops_urls, lodge.sources.accessibility_urls)


def _sources_for_winter_ops(lodge: LodgeItem) -> List[str]:
    return lodge.sources.winner_ops_urls if hasattr(lodge.sources, "winner_ops_urls") else lodge.sources.winter_ops_urls


def _sources_for_accessibility(lodge: LodgeItem) -> List[str]:
    return lodge.sources.accessibility_urls


def _sources_for_access_method(lodge: LodgeItem) -> List[str]:
    return _merge_urls(lodge.sources.access_method_urls, lodge.sources.winter_ops_urls)


def _min_stay_provided(minimum_stay: Optional[str]) -> bool:
    if not _non_empty(minimum_stay):
        return False
    val = (minimum_stay or "").strip().lower()
    return True if val else False  # any explicit text counts, including "not specified"/"not applicable"


# --------------------------------------------------------------------------- #
# Verification logic per lodge                                                #
# --------------------------------------------------------------------------- #
async def verify_lodge(
    evaluator: Evaluator,
    parent_node,
    lodge: LodgeItem,
    lodge_index: int,
) -> None:
    """
    Build verification nodes for a single lodge and run checks.
    """
    # Parent lodge node (parallel, non-critical to allow partial credit per lodge)
    lodge_node = evaluator.add_parallel(
        id=f"lodge_{lodge_index + 1}",
        desc=f"Lodge {lodge_index + 1} (first qualifying lodge) — eligibility + required fields" if lodge_index == 0 else (
             f"Lodge {lodge_index + 1} (second qualifying lodge) — eligibility + required fields" if lodge_index == 1 else
             f"Lodge {lodge_index + 1} (third qualifying lodge) — eligibility + required fields"
        ),
        parent=parent_node,
        critical=False
    )

    # 1) Identity provided (critical existence)
    evaluator.add_custom_node(
        result=_non_empty(lodge.lodge_name) and _non_empty(lodge.national_park),
        id=f"lodge_{lodge_index + 1}_identity",
        desc="Lodge name and national park location are provided",
        parent=lodge_node,
        critical=True,
    )

    # 2) In-park boundary (critical, verify via sources if available)
    in_park_leaf = evaluator.add_leaf(
        id=f"lodge_{lodge_index + 1}_in_park_boundary",
        desc="Lodge is located within the U.S. National Park boundary (not outside the park)",
        parent=lodge_node,
        critical=True,
    )
    park_name = lodge.national_park or "the stated national park"
    claim_boundary = f"The lodge '{lodge.lodge_name or 'UNKNOWN'}' is located within {park_name} boundary (in-park property)."
    await evaluator.verify(
        claim=claim_boundary,
        node=in_park_leaf,
        sources=_sources_for_boundary(lodge),
        additional_instruction=(
            "Confirm the lodge is inside official NPS boundaries (in-park). "
            "Evidence may include NPS pages or concessioner pages explicitly stating 'in the park' or similar. "
            "If sources show it's outside the park (gateway town), mark as incorrect."
        )
    )

    # 3) Winter operation (critical, verify open Dec 2025 – Feb 2026)
    winter_ops_leaf = evaluator.add_leaf(
        id=f"lodge_{lodge_index + 1}_winter_operation",
        desc="Winter operating dates are provided and show the lodge is open and accepting overnight guests during Dec 2025 through Feb 2026",
        parent=lodge_node,
        critical=True,
    )
    claim_winter_ops = (
        "This lodge operates and accepts overnight guests during December 2025, January 2026, and February 2026 "
        "(i.e., the winter 2025–2026 season covers these months)."
    )
    await evaluator.verify(
        claim=claim_winter_ops,
        node=winter_ops_leaf,
        sources=_sources_for_winter_ops(lodge),
        additional_instruction=(
            "Look for winter season schedules or operating dates that explicitly include Dec 2025, Jan 2026, and Feb 2026. "
            "If dates are specified by a range (e.g., 'Dec 15, 2025–Feb 28, 2026'), that's sufficient. "
            "Confirm overnight lodging is available during this period."
        )
    )

    # 4) Accessibility features (critical parent, parallel children)
    acc_parent = evaluator.add_parallel(
        id=f"lodge_{lodge_index + 1}_accessibility_features",
        desc="Accessible room features are confirmed with specific details",
        parent=lodge_node,
        critical=True
    )

    # 4.1 ADA accessible rooms
    ada_leaf = evaluator.add_leaf(
        id=f"lodge_{lodge_index + 1}_ada_accessible_rooms",
        desc="ADA-compliant accessible guest rooms are offered",
        parent=acc_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The lodge '{lodge.lodge_name or 'UNKNOWN'}' offers ADA-compliant accessible guest rooms.",
        node=ada_leaf,
        sources=_sources_for_accessibility(lodge),
        additional_instruction="Verify the presence of ADA-compliant accessible guest rooms on the cited page(s)."
    )

    # 4.2 Roll-in shower
    rollin_leaf = evaluator.add_leaf(
        id=f"lodge_{lodge_index + 1}_roll_in_shower",
        desc="Accessible rooms include roll-in shower facilities",
        parent=acc_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The accessible rooms at '{lodge.lodge_name or 'UNKNOWN'}' include roll-in shower facilities.",
        node=rollin_leaf,
        sources=_sources_for_accessibility(lodge),
        additional_instruction="Look for 'roll-in shower' or equivalent wording clearly describing accessible shower with no step/lip."
    )

    # 4.3 Doorway width
    door_leaf = evaluator.add_leaf(
        id=f"lodge_{lodge_index + 1}_doorway_32in",
        desc="Accessible rooms have doorways with minimum 32-inch clear width",
        parent=acc_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"Accessible guest room doorways at '{lodge.lodge_name or 'UNKNOWN'}' have at least 32 inches of clear width.",
        node=door_leaf,
        sources=_sources_for_accessibility(lodge),
        additional_instruction="Accept if a page states 32-inch minimum, or any doorway width ≥ 32 inches for accessible rooms."
    )

    # 4.4 Visual alarms
    visual_leaf = evaluator.add_leaf(
        id=f"lodge_{lodge_index + 1}_visual_alarms",
        desc="Accessible rooms include visual fire alarm systems for hearing-impaired guests",
        parent=acc_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"Accessible rooms at '{lodge.lodge_name or 'UNKNOWN'}' include visual fire alarm systems (e.g., strobe) for hearing-impaired guests.",
        node=visual_leaf,
        sources=_sources_for_accessibility(lodge),
        additional_instruction="Look for terms like 'visual alarms', 'strobe alarms', or wording indicating visual fire alarm systems in accessible rooms."
    )

    # 4.5 Grab bars
    grab_leaf = evaluator.add_leaf(
        id=f"lodge_{lodge_index + 1}_grab_bars",
        desc="Accessible bathrooms include grab bars in toilet areas",
        parent=acc_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"Accessible bathrooms at '{lodge.lodge_name or 'UNKNOWN'}' include grab bars in toilet areas.",
        node=grab_leaf,
        sources=_sources_for_accessibility(lodge),
        additional_instruction="Verify mention of grab bars in toilet areas or equivalent ADA bathroom fixtures."
    )

    # 5) Winter vehicle access method (critical)
    access_method_leaf = evaluator.add_leaf(
        id=f"lodge_{lodge_index + 1}_winter_vehicle_access_method",
        desc="Winter vehicle access method is specified (e.g., personal vehicle vs. over-snow transportation)",
        parent=lodge_node,
        critical=True
    )
    method_text = lodge.winter_access_method or "UNKNOWN"
    claim_access_method = (
        f"During winter 2025–2026, the primary access method to '{lodge.lodge_name or 'UNKNOWN'}' is '{method_text}'."
    )
    await evaluator.verify(
        claim=claim_access_method,
        node=access_method_leaf,
        sources=_sources_for_access_method(lodge),
        additional_instruction=(
            "Confirm the stated winter access method (e.g., personal vehicle via plowed roads, snowcoach/oversnow transport, shuttle). "
            "Accept reasonable variants (e.g., 'snowcoach' as 'over-snow transportation')."
        )
    )

    # 6) Booking contact for accessible rooms (critical existence)
    evaluator.add_custom_node(
        result=_non_empty(lodge.booking_contact),
        id=f"lodge_{lodge_index + 1}_booking_contact",
        desc="Booking contact information/method for accessible room reservations is provided",
        parent=lodge_node,
        critical=True
    )

    # 7) Minimum stay requirement (non-critical existence)
    evaluator.add_custom_node(
        result=_min_stay_provided(lodge.minimum_stay),
        id=f"lodge_{lodge_index + 1}_minimum_stay_if_applicable",
        desc="Minimum stay requirements are stated if applicable (or explicitly noted as not specified/not applicable)",
        parent=lodge_node,
        critical=False
    )

    # 8) Reference URLs (critical parent)
    ref_parent = evaluator.add_parallel(
        id=f"lodge_{lodge_index + 1}_reference_urls",
        desc="Reference URL(s) provided documenting winter operations and accessibility features",
        parent=lodge_node,
        critical=True
    )

    # 8.1 Winter ops URL supports claim
    winter_url_leaf = evaluator.add_leaf(
        id=f"lodge_{lodge_index + 1}_winter_ops_url",
        desc="At least one reference URL supports winter operation/operating dates",
        parent=ref_parent,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The lodge operates and accepts overnight guests during December 2025, January 2026, and February 2026 "
            "as documented by at least one of the provided winter operation URLs."
        ),
        node=winter_url_leaf,
        sources=_sources_for_winter_ops(lodge),
        additional_instruction="Use multi-URL verification: pass if any one URL explicitly supports the winter operation claim."
    )

    # 8.2 Accessibility URL supports features claim (aggregate claim)
    acc_url_leaf = evaluator.add_leaf(
        id=f"lodge_{lodge_index + 1}_accessibility_url",
        desc="At least one reference URL supports the required accessibility feature claims",
        parent=ref_parent,
        critical=True
    )
    combined_acc_claim = (
        f"The lodge '{lodge.lodge_name or 'UNKNOWN'}' offers ADA-compliant accessible rooms that include: "
        "roll-in shower, doorway clear width of at least 32 inches, visual fire alarm systems, and toilet-area grab bars—"
        "supported by at least one of the provided accessibility URLs."
    )
    await evaluator.verify(
        claim=combined_acc_claim,
        node=acc_url_leaf,
        sources=_sources_for_accessibility(lodge),
        additional_instruction=(
            "Use multi-URL verification: pass if any one URL supports the aggregate accessibility features claim. "
            "Minor wording variations are acceptable (e.g., 'strobe' for visual alarms)."
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
    """
    Evaluate an answer for the National Park winter-accessible lodges task.
    """
    # Initialize evaluator (root is non-critical by framework design; set parallel aggregation)
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

    # Extract lodges from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_lodges(),
        template_class=LodgesExtraction,
        extraction_name="lodges_extraction",
    )

    # Record simple stats
    evaluator.add_custom_info(
        info={"extracted_lodges_count": len(extracted.lodges)},
        info_type="extraction_stats",
        info_name="lodges_count"
    )

    # Ensure exactly 3 lodges (pad with empty objects if fewer; take first 3 if more)
    lodges: List[LodgeItem] = list(extracted.lodges[:3])
    while len(lodges) < 3:
        lodges.append(LodgeItem())

    # Build verification tree per lodge
    for idx, lodge in enumerate(lodges):
        await verify_lodge(evaluator, root, lodge, idx)

    # Return summary
    return evaluator.get_summary()