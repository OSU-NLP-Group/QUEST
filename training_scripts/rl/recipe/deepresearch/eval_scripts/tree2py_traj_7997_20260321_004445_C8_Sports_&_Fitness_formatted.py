import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nba_arenas_capacity_18k_21k"
TASK_DESCRIPTION = (
    "Identify at least 4 NBA arenas located in the United States that have a basketball seating capacity between "
    "18,000 and 21,000 seats. For each arena, provide the following information: the official arena name, the exact "
    "basketball seating capacity, the home NBA team that plays there, the city and state where it is located, the year "
    "the arena opened, and reference URLs that verify this information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ArenaEntry(BaseModel):
    name: Optional[str] = None
    basketball_capacity: Optional[str] = None  # Keep as string for robust extraction; we'll parse numerically.
    home_team: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    opening_year: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ArenasExtraction(BaseModel):
    arenas: List[ArenaEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_arenas() -> str:
    return """
Extract up to 5 NBA arenas mentioned in the answer that meet the task intent. For each arena, extract:
- name: The official arena name (string).
- basketball_capacity: The exact basketball seating capacity as given in the answer (string; do not compute).
- home_team: The full name of the NBA team that plays its home games there (string).
- city: The city where the arena is located (string).
- state: The state (or District of Columbia) where the arena is located (string).
- opening_year: The year the arena opened (string as shown; do not infer).
- reference_urls: A list of URLs explicitly cited in the answer that can verify the above fields. Include only valid, fully qualified URLs actually present in the answer. If none are present, return an empty list.

Rules:
- Only extract what is explicitly present in the answer text.
- If a field is missing, set it to null (or an empty list for reference_urls).
- Prefer the basketball seating capacity (not concert or maximum capacity). If multiple capacities are given, pick the one explicitly labeled for basketball.
- Return a JSON object with a single field "arenas" which is an array (up to 5) of the arena objects described above.
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def parse_capacity_to_int(raw: Optional[str]) -> Optional[int]:
    """Parse a basketball capacity string into an integer if possible."""
    if not raw:
        return None
    s = raw
    # Normalize separators
    s = s.replace(",", " ").replace("\u2009", " ").replace("\xa0", " ")
    # Find integer-like tokens
    nums = re.findall(r"\d{2,6}", s)
    if not nums:
        return None
    # Convert and pick a plausible basketball capacity
    candidates = []
    for token in nums:
        try:
            val = int(token)
            # Heuristic plausible bounds for arena capacities
            if 10000 <= val <= 70000:
                candidates.append(val)
        except Exception:
            continue
    if not candidates:
        return None
    # Prefer a number that falls within 18k–21k if available
    for v in candidates:
        if 18000 <= v <= 21000:
            return v
    # Otherwise choose the largest plausible capacity (often basketball capacity is the larger number in text)
    return max(candidates)


def norm_arena_key(name: Optional[str]) -> str:
    """Normalize arena name for duplicate checks."""
    if not name:
        return ""
    s = name.lower().strip()
    # Minor normalization
    s = s.replace("&", "and")
    s = s.replace("centre", "center")
    # Remove non-alphanumeric except spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Per-arena verification                                                      #
# --------------------------------------------------------------------------- #
async def verify_arena(
    evaluator: Evaluator,
    parent_group: VerificationNode,
    arena: ArenaEntry,
    slot_idx: int,
) -> Tuple[bool, str]:
    """
    Build and run verification for a single arena slot.
    Returns (qualifies, normalized_arena_name_key).
    """
    # Sequential node for this slot
    arena_seq = evaluator.add_sequential(
        id=f"Arena_{slot_idx}",
        desc=f"Arena entry slot #{slot_idx} (if provided) must satisfy all per-arena requirements.",
        parent=parent_group,
        critical=False,  # Non-critical per slot; global constraints handle 'at least 4'
    )

    # 1) Entry present (critical for this slot's sequence)
    entry_present = evaluator.add_custom_node(
        result=non_empty(arena.name),
        id=f"Arena_{slot_idx}_Entry_Present",
        desc="An arena entry is provided in this slot.",
        parent=arena_seq,
        critical=True,
    )

    # 2) Arena details (parallel, all critical)
    details = evaluator.add_parallel(
        id=f"Arena_{slot_idx}_Arena_Details",
        desc="Provided arena entry satisfies all required qualification constraints and includes all required fields with verifying URLs.",
        parent=arena_seq,
        critical=True,
    )

    # Existence and format checks (custom, critical)
    name_provided = evaluator.add_custom_node(
        result=non_empty(arena.name),
        id=f"Arena_{slot_idx}_Arena_Name_Provided",
        desc="Official arena name is provided.",
        parent=details,
        critical=True,
    )

    team_provided = evaluator.add_custom_node(
        result=non_empty(arena.home_team),
        id=f"Arena_{slot_idx}_Home_NBA_Team_Provided",
        desc="Home NBA team name is provided.",
        parent=details,
        critical=True,
    )

    city_provided = evaluator.add_custom_node(
        result=non_empty(arena.city),
        id=f"Arena_{slot_idx}_City_Provided",
        desc="City is provided.",
        parent=details,
        critical=True,
    )

    state_provided = evaluator.add_custom_node(
        result=non_empty(arena.state),
        id=f"Arena_{slot_idx}_State_Provided",
        desc="State (or district, if applicable) is provided.",
        parent=details,
        critical=True,
    )

    opening_year_provided = evaluator.add_custom_node(
        result=non_empty(arena.opening_year),
        id=f"Arena_{slot_idx}_Opening_Year_Provided",
        desc="Year the arena opened is provided.",
        parent=details,
        critical=True,
    )

    cap_int = parse_capacity_to_int(arena.basketball_capacity)
    cap_numeric = evaluator.add_custom_node(
        result=cap_int is not None,
        id=f"Arena_{slot_idx}_Basketball_Capacity_Provided_Numeric",
        desc="Exact basketball seating capacity is provided as a numeric value.",
        parent=details,
        critical=True,
    )

    cap_in_range = evaluator.add_custom_node(
        result=(cap_int is not None and 18000 <= cap_int <= 21000),
        id=f"Arena_{slot_idx}_Basketball_Capacity_In_Range",
        desc="Basketball seating capacity falls within the stated range (18,000–21,000).",
        parent=details,
        critical=True,
    )

    # References presence (critical)
    sources = list(dict.fromkeys(arena.reference_urls or []))  # dedupe while preserving order
    refs_present = evaluator.add_custom_node(
        result=len(sources) > 0,
        id=f"Arena_{slot_idx}_Reference_URLs_Present",
        desc="At least one reference URL is provided to verify claims.",
        parent=details,
        critical=True,
    )

    # Evidence-based checks (each critical, must be supported by provided URLs)
    # 2.1 Current home venue of NBA team
    home_venue_leaf = evaluator.add_leaf(
        id=f"Arena_{slot_idx}_Arena_Is_Current_NBA_Home_Venue",
        desc="Arena is a current home venue for an NBA team.",
        parent=details,
        critical=True,
    )
    if non_empty(arena.home_team) and non_empty(arena.name):
        claim_home = (
            f"The {arena.home_team} are an NBA team and currently play their home games at {arena.name}."
        )
    else:
        claim_home = "The stated NBA team currently plays its home games at the stated arena."
    await evaluator.verify(
        claim=claim_home,
        node=home_venue_leaf,
        sources=sources,
        extra_prerequisites=[refs_present],
        additional_instruction=(
            "Check the provided URLs to confirm that the specified NBA team plays its home games at the named arena. "
            "Accept authoritative sources (team site, NBA.com, arena/operator, Wikipedia). Minor name variations are fine."
        ),
    )

    # 2.2 Located in United States (location verification with sources)
    us_location_leaf = evaluator.add_leaf(
        id=f"Arena_{slot_idx}_Located_In_United_States",
        desc="Arena is located within the United States.",
        parent=details,
        critical=True,
    )
    if non_empty(arena.city) and non_empty(arena.state) and non_empty(arena.name):
        claim_loc = f"{arena.name} is located in {arena.city}, {arena.state}, United States."
    else:
        claim_loc = "The arena is located in the United States (city and state as stated in the answer)."
    await evaluator.verify(
        claim=claim_loc,
        node=us_location_leaf,
        sources=sources,
        extra_prerequisites=[refs_present],
        additional_instruction=(
            "Confirm the arena's location is in the United States. If the page shows the city and state, that suffices. "
            "District of Columbia counts as the United States."
        ),
    )

    # 2.3 Reference URLs verify each specific claim (split into sub-checks for robust evidence grounding)
    refs_all = evaluator.add_parallel(
        id=f"Arena_{slot_idx}_Reference_URLs_Verify_Claims",
        desc="Reference URL(s) verify arena name, capacity, home team, location, and opening year.",
        parent=details,
        critical=True,
    )

    # 2.3.a Name support
    ref_name = evaluator.add_leaf(
        id=f"Arena_{slot_idx}_Ref_Verify_Name",
        desc="Reference confirms the official arena name.",
        parent=refs_all,
        critical=True,
    )
    if non_empty(arena.name):
        claim_name = f"The official name of the arena is '{arena.name}'."
    else:
        claim_name = "The official arena name stated in the answer is correct."
    await evaluator.verify(
        claim=claim_name,
        node=ref_name,
        sources=sources,
        extra_prerequisites=[refs_present],
        additional_instruction="Verify the arena's official name. Minor punctuation or branding variations are acceptable.",
    )

    # 2.3.b Capacity support (basketball)
    ref_capacity = evaluator.add_leaf(
        id=f"Arena_{slot_idx}_Ref_Verify_Basketball_Capacity",
        desc="Reference confirms the exact basketball seating capacity.",
        parent=refs_all,
        critical=True,
    )
    if cap_int is not None and non_empty(arena.name):
        claim_cap = f"The basketball seating capacity of {arena.name} is {cap_int}."
    elif cap_int is not None:
        claim_cap = f"The basketball seating capacity is {cap_int}."
    else:
        claim_cap = "The basketball seating capacity number stated is correct."
    await evaluator.verify(
        claim=claim_cap,
        node=ref_capacity,
        sources=sources,
        extra_prerequisites=[refs_present],
        additional_instruction=(
            "Verify the 'basketball' configuration capacity. If multiple capacities exist, prefer the one explicitly "
            "labeled for basketball. Allow very small variations due to updates (≈±150) if clearly the same metric."
        ),
    )

    # 2.3.c Team support
    ref_team = evaluator.add_leaf(
        id=f"Arena_{slot_idx}_Ref_Verify_Home_Team",
        desc="Reference confirms the home NBA team.",
        parent=refs_all,
        critical=True,
    )
    if non_empty(arena.name) and non_empty(arena.home_team):
        claim_team = f"The home NBA team for {arena.name} is the {arena.home_team}."
    else:
        claim_team = "The stated home NBA team for the arena is correct."
    await evaluator.verify(
        claim=claim_team,
        node=ref_team,
        sources=sources,
        extra_prerequisites=[refs_present],
        additional_instruction="Confirm the listed NBA team plays home games at the arena.",
    )

    # 2.3.d Location support
    ref_location = evaluator.add_leaf(
        id=f"Arena_{slot_idx}_Ref_Verify_Location",
        desc="Reference confirms the city and state (United States).",
        parent=refs_all,
        critical=True,
    )
    if non_empty(arena.name) and non_empty(arena.city) and non_empty(arena.state):
        claim_location = f"The arena {arena.name} is located in {arena.city}, {arena.state}, United States."
    else:
        claim_location = "The stated city and state location (in the United States) for the arena is correct."
    await evaluator.verify(
        claim=claim_location,
        node=ref_location,
        sources=sources,
        extra_prerequisites=[refs_present],
        additional_instruction="Verify the city and state for the arena and that it is in the United States.",
    )

    # 2.3.e Opening year support
    ref_open_year = evaluator.add_leaf(
        id=f"Arena_{slot_idx}_Ref_Verify_Opening_Year",
        desc="Reference confirms the opening year.",
        parent=refs_all,
        critical=True,
    )
    if non_empty(arena.name) and non_empty(arena.opening_year):
        claim_year = f"The arena {arena.name} opened in {arena.opening_year}."
    elif non_empty(arena.opening_year):
        claim_year = f"The arena opened in {arena.opening_year}."
    else:
        claim_year = "The stated opening year of the arena is correct."
    await evaluator.verify(
        claim=claim_year,
        node=ref_open_year,
        sources=sources,
        extra_prerequisites=[refs_present],
        additional_instruction="Confirm the opening year of the arena from the reference URLs.",
    )

    # Determine qualification based on all critical checks within this slot
    critical_nodes = [
        entry_present,
        name_provided,
        team_provided,
        city_provided,
        state_provided,
        opening_year_provided,
        cap_numeric,
        cap_in_range,
        refs_present,
        home_venue_leaf,
        us_location_leaf,
        ref_name,
        ref_capacity,
        ref_team,
        ref_location,
        ref_open_year,
    ]
    qualifies = all(n.status == "passed" for n in critical_nodes)
    return qualifies, norm_arena_key(arena.name)


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
    Evaluate an answer for the NBA arenas capacity task.
    """
    # Initialize evaluator (root is non-critical by design)
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

    # Top-level grouping node reflecting the rubric (set non-critical to allow partial credit on per-arena checks)
    # Note: The original JSON marks this as critical, but our framework enforces that critical parents must have
    # all-critical children. To preserve partial credit per arena, we keep this non-critical and move global critical
    # requirements into a dedicated critical child group below.
    task_group = evaluator.add_parallel(
        id="Find_Qualifying_NBA_Arenas",
        desc="Identify at least 4 NBA arenas located in the United States with basketball seating capacity between 18,000 and 21,000 seats, and provide the required fields with verifying URLs for each.",
        parent=root,
        critical=False,
    )

    # Sub-group for the five arena slots (non-critical group)
    arenas_group = evaluator.add_parallel(
        id="Arena_Entries",
        desc="Arena entries verification group (up to 5 slots).",
        parent=task_group,
        critical=False,
    )

    # Extract up to 5 arena entries from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_arenas(),
        template_class=ArenasExtraction,
        extraction_name="arenas_extraction",
    )

    arenas: List[ArenaEntry] = list(extraction.arenas or [])
    # Ensure exactly 5 slots (pad if needed)
    while len(arenas) < 5:
        arenas.append(ArenaEntry())
    arenas = arenas[:5]

    # Verify each arena slot
    qualifies_flags: List[bool] = []
    qual_names: List[str] = []

    for idx, arena in enumerate(arenas, start=1):
        qualifies, name_key = await verify_arena(
            evaluator=evaluator,
            parent_group=arenas_group,
            arena=arena,
            slot_idx=idx,
        )
        qualifies_flags.append(qualifies)
        qual_names.append(name_key)

    # Global constraints group (critical)
    global_constraints = evaluator.add_parallel(
        id="Global_Constraints",
        desc="Global constraints: at least 4 qualifying arenas and all counted arenas are distinct.",
        parent=task_group,
        critical=True,
    )

    # Compute how many qualify based on per-arena critical checks
    qualifying_indices = [i for i, q in enumerate(qualifies_flags) if q]
    num_qualifying = len(qualifying_indices)

    at_least_4 = evaluator.add_custom_node(
        result=(num_qualifying >= 4),
        id="At_Least_4_Qualifying_Arenas",
        desc="At least 4 of the Arena_1–Arena_5 entries are present and pass all per-arena critical checks.",
        parent=global_constraints,
        critical=True,
    )

    # Distinctness among counted qualifiers (by normalized arena name)
    qualified_names = [qual_names[i] for i in qualifying_indices if qual_names[i]]
    distinct_ok = len(qualified_names) == len(set(qualified_names)) and len(qualified_names) == num_qualifying

    distinct_arenas = evaluator.add_custom_node(
        result=distinct_ok,
        id="Distinct_Arenas",
        desc="Arenas counted toward the minimum are distinct (no duplicate arenas).",
        parent=global_constraints,
        critical=True,
    )

    # Record some custom stats
    evaluator.add_custom_info(
        info={
            "extracted_arenas_count": len(extraction.arenas or []),
            "num_qualifying": num_qualifying,
            "qualifying_slots": [i + 1 for i in qualifying_indices],
            "distinct_ok": distinct_ok,
        },
        info_type="stats",
        info_name="evaluation_stats",
    )

    # Return structured summary
    return evaluator.get_summary()