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
TASK_ID = "nfl_stadiums_capacity_76k_plus_2024_2025"
TASK_DESCRIPTION = (
    "Identify NFL stadiums with official seating capacity ≥ 76,000 as of the 2024–2025 season, and for each included "
    "qualifying stadium provide stadium name, home NFL team(s), exact official seating capacity, and location (city, state)."
)

# Ground truth for the five qualifying stadiums (as of 2024–2025 season)
EXPECTED_STADIUMS: Dict[str, Dict[str, Any]] = {
    "MetLife_Stadium_Entry": {
        "display_name": "MetLife Stadium",
        "keywords": ["metlife stadium", "metlife"],
        "teams": ["New York Giants", "New York Jets"],
        "capacity": 82500,
        "city": "East Rutherford",
        "state_full": "New Jersey",
        "state_abbr": "NJ",
        "node_desc": "MetLife Stadium qualifying entry is provided with all required fields."
    },
    "Lambeau_Field_Entry": {
        "display_name": "Lambeau Field",
        "keywords": ["lambeau field", "lambeau"],
        "teams": ["Green Bay Packers"],
        "capacity": 81441,
        "city": "Green Bay",
        "state_full": "Wisconsin",
        "state_abbr": "WI",
        "node_desc": "Lambeau Field qualifying entry is provided with all required fields."
    },
    "ATT_Stadium_Entry": {
        "display_name": "AT&T Stadium",
        "keywords": ["at&t stadium", "att stadium", "at and t stadium"],
        "teams": ["Dallas Cowboys"],
        "capacity": 80000,
        "city": "Arlington",
        "state_full": "Texas",
        "state_abbr": "TX",
        "node_desc": "AT&T Stadium qualifying entry is provided with all required fields."
    },
    "Arrowhead_Stadium_Entry": {
        "display_name": "Arrowhead Stadium",
        "keywords": ["arrowhead stadium", "geha field at arrowhead stadium", "arrowhead"],
        "teams": ["Kansas City Chiefs"],
        "capacity": 76416,
        "city": "Kansas City",
        "state_full": "Missouri",
        "state_abbr": "MO",
        "node_desc": "Arrowhead Stadium qualifying entry is provided with all required fields."
    },
    "Empower_Field_Entry": {
        "display_name": "Empower Field at Mile High",
        "keywords": ["empower field at mile high", "empower field", "mile high"],
        "teams": ["Denver Broncos"],
        "capacity": 76125,
        "city": "Denver",
        "state_full": "Colorado",
        "state_abbr": "CO",
        "node_desc": "Empower Field at Mile High qualifying entry is provided with all required fields."
    },
}

# Node names for leaves (must match rubric)
LEAF_IDS = {
    "MetLife_Stadium_Entry": {
        "name": "MetLife_Name_Correct",
        "teams": "MetLife_Home_Teams_Correct",
        "capacity": "MetLife_Capacity_Correct",
        "location": "MetLife_Location_City_State_Correct",
    },
    "Lambeau_Field_Entry": {
        "name": "Lambeau_Name_Correct",
        "teams": "Lambeau_Home_Team_Correct",
        "capacity": "Lambeau_Capacity_Correct",
        "location": "Lambeau_Location_City_State_Correct",
    },
    "ATT_Stadium_Entry": {
        "name": "ATT_Name_Correct",
        "teams": "ATT_Home_Team_Correct",
        "capacity": "ATT_Capacity_Correct",
        "location": "ATT_Location_City_State_Correct",
    },
    "Arrowhead_Stadium_Entry": {
        "name": "Arrowhead_Name_Correct",
        "teams": "Arrowhead_Home_Team_Correct",
        "capacity": "Arrowhead_Capacity_Correct",
        "location": "Arrowhead_Location_City_State_Correct",
    },
    "Empower_Field_Entry": {
        "name": "Empower_Name_Correct",
        "teams": "Empower_Home_Team_Correct",
        "capacity": "Empower_Capacity_Correct",
        "location": "Empower_Location_City_State_Correct",
    },
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StadiumItem(BaseModel):
    name: Optional[str] = None
    home_teams: List[str] = Field(default_factory=list)
    capacity: Optional[str] = None  # keep as string for robustness
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class StadiumExtraction(BaseModel):
    stadiums: List[StadiumItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stadiums() -> str:
    return """
    Extract all NFL stadium entries mentioned in the answer that the answer itself claims have an official seating capacity of 76,000 or greater as of the 2024–2025 season.
    For each qualifying stadium mentioned in the answer, extract an object with:
    - name: The stadium name exactly as written in the answer (string).
    - home_teams: A list of the NFL home team names for that stadium, as provided in the answer (array of strings).
    - capacity: The exact official seating capacity stated in the answer (string; include commas if present; if the answer includes extra text like 'expandable to', keep the full capacity phrase but ensure the official base capacity number at the start is preserved).
    - location_city: The city name for the stadium location, as stated in the answer (string). If the answer provides 'City, State', split them accordingly.
    - location_state: The state for the stadium location, as stated in the answer (string; can be two-letter abbreviation like 'NJ' or full name like 'New Jersey').
    - source_urls: A list of any URLs (if any) cited in the answer specifically for that stadium (array of strings). If none are included, return an empty array.
    
    Return a JSON object with a single field:
    {
      "stadiums": [ ... array of stadium objects as above ... ]
    }
    
    IMPORTANT:
    - Only extract entries that the answer claims have capacity ≥ 76,000.
    - Do not invent fields not present in the answer text.
    - If any field is missing for a stadium, set it to null (or empty list for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
STATE_ABBR_TO_FULL = {
    "NJ": "New Jersey",
    "WI": "Wisconsin",
    "TX": "Texas",
    "MO": "Missouri",
    "CO": "Colorado",
}

def _normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _name_matches(candidate: Optional[str], expected_display: str, keywords: List[str]) -> bool:
    cand = _normalize_text(candidate)
    if not cand:
        return False
    exp = _normalize_text(expected_display)
    if exp and exp in cand:
        return True
    for kw in keywords:
        nkw = _normalize_text(kw)
        if nkw and nkw in cand:
            return True
    return False


def _canonical_team(name: str) -> Optional[str]:
    n = _normalize_text(name)
    # New York Giants
    if "giant" in n:
        return "New York Giants"
    # New York Jets
    if "jet" in n:
        return "New York Jets"
    # Green Bay Packers
    if "packer" in n:
        return "Green Bay Packers"
    # Dallas Cowboys
    if "cowboy" in n:
        return "Dallas Cowboys"
    # Kansas City Chiefs
    if "chief" in n:
        return "Kansas City Chiefs"
    # Denver Broncos
    if "bronco" in n:
        return "Denver Broncos"
    return None


def _teams_match(extracted: List[str], expected: List[str]) -> bool:
    extracted_canon = set()
    for t in extracted:
        ct = _canonical_team(t)
        if ct:
            extracted_canon.add(ct)
    expected_set = set(expected)
    return extracted_canon == expected_set


def _first_int_from_text(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    # find the first group of digits (potentially with commas)
    m = re.search(r"(\d[\d,]*)", s)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        return int(num)
    except ValueError:
        return None


def _state_to_full(state_str: Optional[str]) -> Optional[str]:
    if not state_str:
        return None
    s = state_str.strip()
    # Try abbreviation first (case-insensitive)
    abbr = s.upper()
    if abbr in STATE_ABBR_TO_FULL:
        return STATE_ABBR_TO_FULL[abbr]
    # Otherwise try match full names roughly
    normalized = _normalize_text(s)
    for full in STATE_ABBR_TO_FULL.values():
        if _normalize_text(full) == normalized:
            return full
    # If not recognized, return the original trimmed state
    return s


def _city_matches(extracted_city: Optional[str], expected_city: str) -> bool:
    return _normalize_text(extracted_city) == _normalize_text(expected_city)


def _state_matches(extracted_state: Optional[str], expected_full: str, expected_abbr: str) -> bool:
    if not extracted_state:
        return False
    ex_full = _state_to_full(extracted_state)
    if not ex_full:
        return False
    return _normalize_text(ex_full) == _normalize_text(expected_full) or extracted_state.strip().upper() == expected_abbr.upper()


def _find_best_entry(extraction: StadiumExtraction, keywords: List[str]) -> Tuple[Optional[StadiumItem], Optional[int]]:
    """
    Find the best matching stadium entry from extraction by keywords in the name field.
    Returns (StadiumItem or None, index or None).
    """
    best_idx = None
    best_score = 0
    for idx, item in enumerate(extraction.stadiums):
        cand = _normalize_text(item.name)
        if not cand:
            continue
        score = 0
        for kw in keywords:
            nkw = _normalize_text(kw)
            if nkw and nkw in cand:
                score += 1
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is None or best_score == 0:
        return None, None
    return extraction.stadiums[best_idx], best_idx


# --------------------------------------------------------------------------- #
# Verification (tree construction)                                            #
# --------------------------------------------------------------------------- #
async def _add_stadium_entry_checks(
    evaluator: Evaluator,
    parent_node,
    entry_node_id: str,
    entry_node_desc: str,
    leaves_ids: Dict[str, str],
    expected: Dict[str, Any],
    extraction: StadiumExtraction,
    debug_bucket: Dict[str, Any],
) -> None:
    """
    Build the per-stadium verification subtree:
    - Parent: a parallel node for the stadium entry (non-critical).
    - Children: four critical leaves (custom boolean checks).
    """
    # Add parent node for this stadium entry
    node = evaluator.add_parallel(
        id=entry_node_id,
        desc=entry_node_desc,
        parent=parent_node,
        critical=False
    )

    # Find the best matching extracted stadium entry
    matched_item, matched_index = _find_best_entry(extraction, expected["keywords"])

    # For debugging info
    dbg = {
        "matched_index": matched_index,
        "extracted": matched_item.dict() if matched_item else None
    }
    debug_bucket[entry_node_id] = dbg

    # 1) Name Correct
    name_ok = _name_matches(matched_item.name if matched_item else None, expected["display_name"], expected["keywords"])
    evaluator.add_custom_node(
        result=name_ok,
        id=leaves_ids["name"],
        desc=f"Stadium name is provided as {expected['display_name']}.",
        parent=node,
        critical=True
    )

    # 2) Home NFL Team(s) Correct
    teams_ok = False
    if matched_item:
        teams_ok = _teams_match(matched_item.home_teams, expected["teams"])
    evaluator.add_custom_node(
        result=teams_ok,
        id=leaves_ids["teams"],
        desc=f"Home NFL team(s) match the constraints: {', '.join(expected['teams'])}.",
        parent=node,
        critical=True
    )

    # 3) Exact Official Seating Capacity Correct
    capacity_ok = False
    if matched_item:
        first_int = _first_int_from_text(matched_item.capacity)
        capacity_ok = (first_int == expected["capacity"])
    evaluator.add_custom_node(
        result=capacity_ok,
        id=leaves_ids["capacity"],
        desc=f"Exact official seating capacity matches the constraints for 2024–2025: {expected['capacity']:,}.",
        parent=node,
        critical=True
    )

    # 4) Location City/State Correct
    location_ok = False
    if matched_item:
        city_ok = _city_matches(matched_item.location_city, expected["city"])
        state_ok = _state_matches(matched_item.location_state, expected["state_full"], expected["state_abbr"])
        location_ok = (city_ok and state_ok)
    evaluator.add_custom_node(
        result=location_ok,
        id=leaves_ids["location"],
        desc=f"Location is provided as a city and state and is correct for {expected['display_name']}.",
        parent=node,
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the NFL stadiums (capacity ≥ 76,000) task.
    """
    # Initialize evaluator with PARALLEL strategy at root
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

    # Extract structured stadium entries from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_stadiums(),
        template_class=StadiumExtraction,
        extraction_name="stadium_extraction"
    )

    # Ground truth info for transparency
    gt_list = []
    for node_id, exp in EXPECTED_STADIUMS.items():
        gt_list.append({
            "node_id": node_id,
            "display_name": exp["display_name"],
            "home_teams": exp["teams"],
            "capacity": exp["capacity"],
            "city": exp["city"],
            "state": exp["state_full"],
            "state_abbr": exp["state_abbr"],
        })
    evaluator.add_ground_truth({"expected_stadiums": gt_list}, gt_type="ground_truth")

    # Optional debug bucket to log which extracted item matched which stadium
    match_debug: Dict[str, Any] = {}

    # Build per-stadium verification subtrees (parallel under root)
    for entry_node_id, expected in EXPECTED_STADIUMS.items():
        leaves = LEAF_IDS[entry_node_id]
        await _add_stadium_entry_checks(
            evaluator=evaluator,
            parent_node=root,
            entry_node_id=entry_node_id,
            entry_node_desc=expected["node_desc"],
            leaves_ids=leaves,
            expected=expected,
            extraction=extraction,
            debug_bucket=match_debug
        )

    # Record matching debug info
    evaluator.add_custom_info(match_debug, info_type="matching_debug", info_name="matching_debug")

    # Return the evaluation summary
    return evaluator.get_summary()