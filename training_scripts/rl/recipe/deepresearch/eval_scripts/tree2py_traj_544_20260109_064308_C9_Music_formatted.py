import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_concert_arenas_3"
TASK_DESCRIPTION = (
    "Identify 3 major concert arenas in the United States, each located in a different state, that meet ALL of the following criteria:\n\n"
    "1. Concert seating capacity between 15,000 and 20,000\n"
    "2. Opened or underwent major renovation between January 1, 2018 and December 31, 2024\n"
    "3. Meet ADA federal accessibility requirements for wheelchair-accessible seating (approximately 1% of total capacity)\n"
    "4. Each arena must be in a different U.S. state\n\n"
    "For each arena, provide:\n"
    "- Official venue name\n"
    "- City and state location\n"
    "- Exact concert seating capacity\n"
    "- Year of opening or major renovation completion\n"
    "- Number of ADA wheelchair-accessible spaces\n"
    "- Official source URL that verifies the venue specifications"
)

YEAR_MIN = 2018
YEAR_MAX = 2024

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ArenaItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None  # Keep as text for robustness
    year_open_or_renovation: Optional[str] = None  # Keep as text
    ada_wheelchair_spaces: Optional[str] = None  # Keep as text
    source_url: Optional[str] = None


class ArenasExtraction(BaseModel):
    arenas: List[ArenaItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_arenas() -> str:
    return (
        "Extract all arenas mentioned in the answer that the agent claims satisfy the task. "
        "Return a JSON object with an array field `arenas`, where each item includes the following fields (verbatim from the answer):\n"
        "- name: Official venue name\n"
        "- city: City location\n"
        "- state: State location (full name or abbreviation)\n"
        "- capacity: Exact concert seating capacity as stated in the answer (text as-is)\n"
        "- year_open_or_renovation: Year of opening or major renovation completion as stated (text as-is)\n"
        "- ada_wheelchair_spaces: Number of ADA wheelchair-accessible spaces (text as-is)\n"
        "- source_url: The official or authoritative source URL provided that verifies the venue specifications (full URL)\n\n"
        "Rules:\n"
        "1. Do NOT invent information. Only extract exactly what appears in the answer text.\n"
        "2. If any field is not explicitly provided for an arena, return null for that field.\n"
        "3. If more than 3 arenas are mentioned, include them all in the `arenas` array; the evaluator will select the first 3.\n"
        "4. If URLs are given in markdown, extract the actual URL string.\n"
        "5. Ensure URLs include protocol (http or https). If missing, keep as-is (do not modify)."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty(s: Optional[str]) -> bool:
    return bool(s) and bool(str(s).strip())


def _is_valid_url(url: Optional[str]) -> bool:
    if not _is_nonempty(url):
        return False
    # Basic validation: must contain protocol and a dot
    u = str(url).strip()
    return u.startswith("http://") or u.startswith("https://")


def _parse_int_from_text(text: Optional[str]) -> Optional[int]:
    """
    Extract the first plausible integer from text.
    Handles thousand separators like '18,500' or '18 500'.
    """
    if not _is_nonempty(text):
        return None
    s = str(text)
    # Prefer numbers with thousand separators
    m = re.search(r'\b\d{1,3}(?:[,\s]\d{3})+\b', s)
    if m:
        try:
            return int(re.sub(r'[,\s]', '', m.group(0)))
        except Exception:
            pass
    # Fallback: first 3–6 digit number
    m2 = re.search(r'\b\d{3,6}\b', s)
    if m2:
        try:
            return int(m2.group(0))
        except Exception:
            pass
    # Last resort: any integer
    m3 = re.search(r'\b\d+\b', s)
    if m3:
        try:
            return int(m3.group(0))
        except Exception:
            pass
    return None


def _parse_year(text: Optional[str]) -> Optional[int]:
    if not _is_nonempty(text):
        return None
    m = re.search(r'\b(19|20)\d{2}\b', str(text))
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


def _approx_one_percent(ada_spaces: Optional[int], capacity: Optional[int]) -> bool:
    """
    Check ADA wheelchair-accessible spaces are approximately 1% of total capacity.
    We allow a tolerance range of 0.8% to 1.5% to account for small deviations.
    """
    if not ada_spaces or not capacity or capacity <= 0 or ada_spaces < 0:
        return False
    ratio = ada_spaces / capacity
    return 0.008 <= ratio <= 0.015


# --------------------------------------------------------------------------- #
# Verification logic per arena                                                #
# --------------------------------------------------------------------------- #
async def verify_arena(
    evaluator: Evaluator,
    parent_node,
    arena: ArenaItem,
    index: int
) -> None:
    """
    Build verification sub-tree for a single arena following the rubric.
    """
    # Pre-parse numeric values for custom checks
    capacity_num = _parse_int_from_text(arena.capacity)
    year_num = _parse_year(arena.year_open_or_renovation)
    ada_num = _parse_int_from_text(arena.ada_wheelchair_spaces)

    # Arena node (non-critical to allow partial scoring across arenas)
    arena_node = evaluator.add_parallel(
        id=f"arena_{index+1}",
        desc=f"Arena {index+1} meets all specified criteria",
        parent=parent_node,
        critical=False
    )

    # 1) Basic info (critical)
    basic_node = evaluator.add_parallel(
        id=f"arena_{index+1}_basic_info",
        desc=f"Basic identification information for arena {index+1}",
        parent=arena_node,
        critical=True
    )

    # 1.a Name provided
    evaluator.add_custom_node(
        result=_is_nonempty(arena.name),
        id=f"arena_{index+1}_name",
        desc="Official venue name is provided",
        parent=basic_node,
        critical=True
    )

    # 1.b Location provided (city and state)
    evaluator.add_custom_node(
        result=_is_nonempty(arena.city) and _is_nonempty(arena.state),
        id=f"arena_{index+1}_location",
        desc="City and state location are provided",
        parent=basic_node,
        critical=True
    )

    # 1.c US location (verify against source URL if available)
    us_loc_leaf = evaluator.add_leaf(
        id=f"arena_{index+1}_us_location",
        desc="Arena is located in the United States",
        parent=basic_node,
        critical=True
    )
    # Claim focuses on US location with provided city/state
    loc_city = arena.city or ""
    loc_state = arena.state or ""
    venue_name = arena.name or ""
    claim_us_location = (
        f"The venue '{venue_name}' is located in {loc_city}, {loc_state}, United States."
    )
    await evaluator.verify(
        claim=claim_us_location,
        node=us_loc_leaf,
        sources=arena.source_url,
        additional_instruction=(
            "Verify that the page confirms the venue's location is within the United States, "
            "ideally showing the city and state. Minor formatting or abbreviation differences "
            "are acceptable."
        ),
    )

    # 1.d Reference URL provided
    evaluator.add_custom_node(
        result=_is_valid_url(arena.source_url),
        id=f"arena_{index+1}_reference_url",
        desc="A source URL is provided that verifies the venue specifications",
        parent=basic_node,
        critical=True
    )

    # 2) Capacity (critical)
    capacity_node = evaluator.add_parallel(
        id=f"arena_{index+1}_capacity",
        desc=f"Capacity specifications for arena {index+1}",
        parent=arena_node,
        critical=True
    )

    # 2.a Capacity provided
    evaluator.add_custom_node(
        result=capacity_num is not None,
        id=f"arena_{index+1}_capacity_provided",
        desc="Exact concert seating capacity is provided",
        parent=capacity_node,
        critical=True
    )

    # 2.b Capacity range (15,000–20,000 inclusive)
    evaluator.add_custom_node(
        result=capacity_num is not None and 15000 <= capacity_num <= 20000,
        id=f"arena_{index+1}_capacity_range",
        desc="Concert seating capacity is between 15,000 and 20,000 inclusive",
        parent=capacity_node,
        critical=True
    )

    # 3) Opening or renovation timeframe (critical)
    openren_node = evaluator.add_parallel(
        id=f"arena_{index+1}_opening_or_renovation",
        desc=f"Opening or renovation date information for arena {index+1}",
        parent=arena_node,
        critical=True
    )

    # 3.a Year provided
    evaluator.add_custom_node(
        result=year_num is not None,
        id=f"arena_{index+1}_year_provided",
        desc="Year of opening or major renovation completion is stated",
        parent=openren_node,
        critical=True
    )

    # 3.b Timeframe within 2018–2024 inclusive
    evaluator.add_custom_node(
        result=year_num is not None and YEAR_MIN <= year_num <= YEAR_MAX,
        id=f"arena_{index+1}_timeframe",
        desc="Opening or renovation completion occurred between January 1, 2018 and December 31, 2024",
        parent=openren_node,
        critical=True
    )

    # 4) Accessibility (critical)
    access_node = evaluator.add_parallel(
        id=f"arena_{index+1}_accessibility",
        desc=f"ADA accessibility information for arena {index+1}",
        parent=arena_node,
        critical=True
    )

    # 4.a ADA wheelchair-accessible spaces provided
    evaluator.add_custom_node(
        result=ada_num is not None,
        id=f"arena_{index+1}_ada_count_provided",
        desc="Number of wheelchair-accessible spaces is provided",
        parent=access_node,
        critical=True
    )

    # 4.b ADA requirement approximately 1% of capacity
    evaluator.add_custom_node(
        result=_approx_one_percent(ada_num, capacity_num),
        id=f"arena_{index+1}_ada_requirement",
        desc="Wheelchair-accessible spaces are approximately 1% of total capacity (per stated constraint)",
        parent=access_node,
        critical=True
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
) -> Dict:
    """
    Evaluate an answer for the '3 US concert arenas' task.
    """
    # Initialize evaluator (root node is non-critical by design to allow partial credit)
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

    # Extract structured arenas data
    extracted = await evaluator.extract(
        prompt=prompt_extract_arenas(),
        template_class=ArenasExtraction,
        extraction_name="arenas_extraction",
    )

    # Compute counts based on what the answer actually provided
    provided_arenas = [a for a in extracted.arenas if _is_nonempty(a.name)]
    provided_count = len(provided_arenas)

    # Critical: Exactly 3 arenas are provided
    evaluator.add_custom_node(
        result=(provided_count == 3),
        id="count_three_arenas",
        desc="Exactly 3 arenas are provided",
        parent=root,
        critical=True
    )

    # Select first 3 for downstream verification; pad if fewer
    selected_arenas: List[ArenaItem] = extracted.arenas[:3]
    if len(selected_arenas) < 3:
        selected_arenas += [ArenaItem() for _ in range(3 - len(selected_arenas))]

    # Critical: Distinct states among the 3 selected arenas
    states = [a.state.strip() for a in selected_arenas if _is_nonempty(a.state)]
    distinct_states_ok = (len(states) == 3) and (len(set(states)) == 3)

    evaluator.add_custom_node(
        result=distinct_states_ok,
        id="distinct_states",
        desc="All 3 arenas are located in 3 different U.S. states (no two share the same state)",
        parent=root,
        critical=True
    )

    # Build verification subtrees for each arena
    for i, arena in enumerate(selected_arenas[:3]):
        await verify_arena(evaluator, root, arena, i)

    # Return summary
    return evaluator.get_summary()