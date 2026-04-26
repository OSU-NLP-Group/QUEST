import asyncio
import logging
import re
import calendar
from datetime import date
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "holiday_diy_project_planning"
TASK_DESCRIPTION = """A parent in the United States wants to take their 7-year-old child to a free kids DIY workshop at a major hardware store in February 2026 to build a woodworking project. After the workshop, they plan to use the skills learned to create a holiday wreath together. They need to purchase basic wreath-making supplies on Christmas Eve 2025 (December 24, 2025) after 5:30 PM at a store that carries craft supplies.

Based on the workshop schedules of Home Depot and Lowe's, and considering the Christmas Eve 2025 operating hours of Home Depot, Lowe's, and Hobby Lobby:

1. Which store should they attend for the kids workshop in February 2026, and on what specific date (month, day, and year)?

2. Which store should they visit for purchasing wreath-making supplies on Christmas Eve 2025, and what time does this store close that day?

3. List the four essential categories of supplies needed to make a basic wreath (provide the general supply type, not specific brands)."""

# Workshop rules (per rubric)
WORKSHOP_RULES = {
    "home depot": {"weekday": calendar.SATURDAY, "nth": 1},
    "lowe's": {"weekday": calendar.SATURDAY, "nth": 3},
}

# Christmas Eve 2025 closing times (per rubric)
# Stored as minutes from midnight, 24h clock
CLOSING_TIMES_MIN = {
    "home depot": 17 * 60,      # 5:00 PM
    "lowe's": 18 * 60,          # 6:00 PM
    "hobby lobby": 17 * 60 + 30 # 5:30 PM
}
AFTER_530PM_THRESHOLD_MIN = 17 * 60 + 30  # 5:30 PM

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #

class WorkshopExtraction(BaseModel):
    workshop_store: Optional[str] = None
    workshop_date_text: Optional[str] = None  # e.g., "February 15, 2026"
    # Whether the answer itself explicitly states the 7-year-old meets the age requirement
    age_requirement_indicated: Optional[bool] = None


class ShoppingExtraction(BaseModel):
    shopping_store: Optional[str] = None
    closing_time_text: Optional[str] = None  # e.g., "6 PM", "5:30 pm"


class SuppliesExtraction(BaseModel):
    categories: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12
}

_DOW = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
]


def normalize_store_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower()
    s = s.replace("’", "'")
    # Remove leading "the "
    s = re.sub(r"^\s*the\s+", "", s)
    # Normalize typical variants
    s = s.replace("the home depot", "home depot")
    s = s.replace("lowes", "lowe's")
    s = s.replace("loew's", "lowe's")
    s = s.replace("home-depot", "home depot")
    s = s.replace("hobby-lobby", "hobby lobby")
    s = re.sub(r"\s+", " ", s).strip()
    if s in {"home depot", "lowe's", "hobby lobby"}:
        return s
    # Try coarse matching
    if "depot" in s:
        return "home depot"
    if "lowe" in s:
        return "lowe's"
    if "hobby" in s and "lobby" in s:
        return "hobby lobby"
    return s


def parse_time_to_minutes(t: Optional[str]) -> Optional[int]:
    if not t:
        return None
    raw = t.strip().lower()
    # Remove periods in a.m./p.m.
    raw = raw.replace(".", "")
    # Common patterns:
    # 1) 6 pm | 6pm | 6:00 pm | 6:00pm
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", raw)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        ampm = m.group(3)
        if hour == 12:
            hour = 0
        if ampm == "pm":
            hour += 12
        return hour * 60 + minute

    # 2) 18:00 (24-hour)
    m2 = re.search(r"\b(\d{1,2}):(\d{2})\b", raw)
    if m2:
        hour = int(m2.group(1))
        minute = int(m2.group(2))
        if 0 <= hour < 24:
            return hour * 60 + minute

    # 3) 6pm (without space)
    m3 = re.search(r"\b(\d{1,2})\s*(pm|am)\b", raw)
    if m3:
        hour = int(m3.group(1))
        ampm = m3.group(2)
        minute = 0
        if hour == 12:
            hour = 0
        if ampm == "pm":
            hour += 12
        return hour * 60 + minute

    # 4) Plain hour (ambiguous) — avoid guessing
    return None


def strip_leading_dow(s: str) -> str:
    ss = s.strip()
    # Remove leading day-of-week like "Saturday," or "Sat,"
    m = re.match(r"^\s*([A-Za-z]{3,9})[, ]+\s*(.*)$", ss)
    if m:
        token = m.group(1).lower()
        rest = m.group(2)
        if token in _DOW or token[:3] in [d[:3] for d in _DOW]:
            return rest.strip()
    return ss


def parse_date_text_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    text = strip_leading_dow(s.strip())

    # 1) Month name formats: "February 15, 2026" or "Feb 15, 2026"
    m = re.search(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
        r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?"
        r"(?:,?\s+(\d{4}))\b",
        text,
        flags=re.IGNORECASE
    )
    if m:
        mon_str = m.group(1).lower()
        day = int(m.group(2))
        year = int(m.group(3))
        mon_key = mon_str[:3] if len(mon_str) > 3 else mon_str
        for k, v in _MONTHS.items():
            if k.startswith(mon_key) and len(k) <= 3:
                try:
                    return date(year, v, day)
                except Exception:
                    return None

    # 2) Numeric: 2/15/2026 or 02/15/2026
    m2 = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
    if m2:
        mon = int(m2.group(1))
        day = int(m2.group(2))
        year = int(m2.group(3))
        try:
            return date(year, mon, day)
        except Exception:
            return None

    # 3) ISO: 2026-02-15
    m3 = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
    if m3:
        year = int(m3.group(1))
        mon = int(m3.group(2))
        day = int(m3.group(3))
        try:
            return date(year, mon, day)
        except Exception:
            return None

    return None


def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> Optional[date]:
    # weekday: Monday=0 ... Sunday=6
    # Use calendar.monthcalendar for robustness
    weeks = calendar.monthcalendar(year, month)
    count = 0
    for w in weeks:
        day = w[weekday]
        if day != 0:
            count += 1
            if count == n:
                return date(year, month, day)
    return None


def categories_contains_any(categories: List[str], needles: List[str]) -> bool:
    for item in categories:
        txt = (item or "").lower()
        # Remove punctuation for loose matching
        txt_norm = re.sub(r"[^\w\s\-]", " ", txt)
        for n in needles:
            n_norm = re.sub(r"[^\w\s\-]", " ", n.lower())
            if n_norm in txt_norm:
                return True
    return False


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #

def prompt_extract_workshop() -> str:
    return """
    Extract the chosen store and specific date for the free kids DIY workshop in February 2026, and whether the answer indicates that the 7-year-old meets the age requirement.
    Provide:
    - workshop_store: the one store named for the February 2026 kids workshop (choose one; prefer Home Depot or Lowe's if multiple are mentioned).
    - workshop_date_text: the specific date the answer provides for the February 2026 workshop (e.g., "February 3, 2026" or "Feb 3, 2026" or "2/3/2026"). Return null if not specified.
    - age_requirement_indicated: true if the answer explicitly states or clearly implies the 7-year-old meets the age range for the chosen store’s kids workshop; false otherwise.
    """


def prompt_extract_shopping() -> str:
    return """
    Extract the chosen store and the closing time for shopping on Christmas Eve 2025 (December 24, 2025).
    Provide:
    - shopping_store: the one store selected for buying wreath-making supplies on Dec 24, 2025 (choose one; if multiple listed, pick the one the answer recommends).
    - closing_time_text: the closing time stated for that store on Dec 24, 2025, as mentioned in the answer (e.g., "6 PM", "5:30 pm", "17:30"). Return null if not provided.
    """


def prompt_extract_supplies() -> str:
    return """
    Extract the list of supply categories provided for making a basic wreath. Return general category names as they appear.
    Provide:
    - categories: an array of supply category names (not brands). Keep the original wording from the answer.
    """


# --------------------------------------------------------------------------- #
# Verification logic builders                                                 #
# --------------------------------------------------------------------------- #

async def build_workshop_selection(
    evaluator: Evaluator,
    parent,
    workshop: WorkshopExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Workshop_Selection",
        desc="Identify which store to attend for the kids workshop in February 2026 and provide the specific date, consistent with schedule and age constraints.",
        parent=parent,
        critical=True
    )

    # Normalize
    store_norm = normalize_store_name(workshop.workshop_store or "")
    # Eligible store check
    eligible = store_norm in {"home depot", "lowe's"}
    evaluator.add_custom_node(
        result=eligible,
        id="Workshop_Store_Is_Eligible",
        desc="Chosen workshop store is either Home Depot or Lowe's.",
        parent=node,
        critical=True
    )

    # Date match check: February 2026 and matches schedule rule
    parsed = parse_date_text_to_date(workshop.workshop_date_text or "")
    in_feb_2026 = parsed is not None and parsed.year == 2026 and parsed.month == 2
    schedule_ok = False
    if in_feb_2026 and eligible:
        rule = WORKSHOP_RULES[store_norm]
        expected_date = nth_weekday_of_month(2026, 2, rule["weekday"], rule["nth"])
        schedule_ok = (expected_date is not None and parsed == expected_date)

    evaluator.add_custom_node(
        result=in_feb_2026 and schedule_ok,
        id="Workshop_Date_Matches_Store_Schedule_Feb_2026",
        desc="Provided workshop date (month/day/year) is in February 2026 and matches the chosen store’s schedule rule (Home Depot: first Saturday; Lowe’s: third Saturday).",
        parent=node,
        critical=True
    )

    # Age requirement indicated in the answer (explicitly)
    age_leaf = evaluator.add_leaf(
        id="Age_Requirement_Met",
        desc="Answer indicates the 7-year-old meets the chosen workshop’s age requirement (Home Depot 5–12 or Lowe’s 4–11).",
        parent=node,
        critical=True
    )
    # Use LLM to check the answer context for explicitness
    chosen = workshop.workshop_store or "the chosen workshop"
    claim = f"The answer explicitly indicates that a 7-year-old meets the age requirement for {chosen}'s kids workshop."
    await evaluator.verify(
        claim=claim,
        node=age_leaf,
        additional_instruction="Look for an explicit or clearly implied statement that the 7-year-old qualifies for the selected store's kids workshop age range (Home Depot: ages 5–12; Lowe’s: ages 4–11)."
    )


async def build_christmas_eve_selection(
    evaluator: Evaluator,
    parent,
    shopping: ShoppingExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Christmas_Eve_Store_Selection",
        desc="Identify which store to visit for wreath-making supplies on Dec 24, 2025 after 5:30 PM, and state the closing time.",
        parent=parent,
        critical=True
    )

    store_norm = normalize_store_name(shopping.shopping_store or "")
    in_considered_set = store_norm in {"home depot", "lowe's", "hobby lobby"}
    evaluator.add_custom_node(
        result=in_considered_set,
        id="Store_Is_From_Considered_Set",
        desc="Chosen shopping store is one of: Home Depot, Lowe's, Hobby Lobby.",
        parent=node,
        critical=True
    )

    close_min = parse_time_to_minutes(shopping.closing_time_text or "")
    after_530_ok = (close_min is not None and close_min > AFTER_530PM_THRESHOLD_MIN)
    evaluator.add_custom_node(
        result=after_530_ok,
        id="After_530PM_Feasibility_Satisfied",
        desc="Chosen store’s Christmas Eve 2025 closing time is strictly later than 5:30 PM, satisfying the 'after 5:30 PM' shopping constraint.",
        parent=node,
        critical=True
    )

    matches_given_hours = False
    if in_considered_set and close_min is not None:
        expected = CLOSING_TIMES_MIN[store_norm]
        matches_given_hours = (close_min == expected)

    evaluator.add_custom_node(
        result=matches_given_hours,
        id="Closing_Time_Matches_Given_Hours",
        desc="The closing time stated for the chosen store matches the provided Christmas Eve 2025 hours constraint (Home Depot 5pm, Lowe’s 6pm, Hobby Lobby 5:30pm).",
        parent=node,
        critical=True
    )


async def build_essential_supplies(
    evaluator: Evaluator,
    parent,
    supplies: SuppliesExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Essential_Supplies_List",
        desc="List the four essential categories of supplies needed to make a basic wreath (general types, not brands).",
        parent=parent,
        critical=True
    )

    cats = supplies.categories or []
    cats_clean = [c.strip() for c in cats if c and c.strip()]

    # Provides at least four categories (not fewer)
    evaluator.add_custom_node(
        result=(len(cats_clean) >= 4),
        id="Provides_Four_Categories",
        desc="Answer provides four supply categories (not fewer).",
        parent=node,
        critical=True
    )

    # Includes wreath form/base
    wreath_syns = [
        "wreath form", "wreath base", "wire wreath frame", "wire frame",
        "wreath frame", "grapevine wreath", "foam wreath", "straw wreath",
        "metal wreath frame", "wreath ring"
    ]
    evaluator.add_custom_node(
        result=categories_contains_any(cats_clean, wreath_syns),
        id="Includes_Wreath_Form",
        desc="List includes a wreath form/base.",
        parent=node,
        critical=True
    )

    # Includes deco mesh
    mesh_syns = ["deco mesh", "mesh ribbon", "poly mesh", "deco-mesh", "mesh"]
    evaluator.add_custom_node(
        result=categories_contains_any(cats_clean, mesh_syns),
        id="Includes_Deco_Mesh",
        desc="List includes deco mesh.",
        parent=node,
        critical=True
    )

    # Includes ribbon
    evaluator.add_custom_node(
        result=categories_contains_any(cats_clean, ["ribbon"]),
        id="Includes_Ribbon",
        desc="List includes ribbon.",
        parent=node,
        critical=True
    )

    # Includes fasteners (pipe cleaners, zip ties, floral wire, etc.)
    fastener_syns = [
        "pipe cleaners", "chenille stems", "zip ties", "twist ties",
        "floral wire", "wire ties", "craft wire"
    ]
    evaluator.add_custom_node(
        result=categories_contains_any(cats_clean, fastener_syns),
        id="Includes_Fasteners",
        desc="List includes fasteners (e.g., pipe cleaners, zip ties, or floral wire).",
        parent=node,
        critical=True
    )

    # General types, not brands (LLM-based simple verify)
    general_leaf = evaluator.add_leaf(
        id="General_Types_Not_Brands",
        desc="Supplies are stated as general categories/types rather than specific brands.",
        parent=node,
        critical=True
    )
    claim = f"The listed supply items are general categories (not brand names): {cats_clean}."
    await evaluator.verify(
        claim=claim,
        node=general_leaf,
        additional_instruction="Evaluate if the items read like generic categories (e.g., 'wreath frame', 'deco mesh', 'ribbon', 'floral wire') rather than brand names (e.g., 'Ashland', 'FloraCraft')."
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

    # Compute ground-truth dates for February 2026 schedule rules
    first_sat = nth_weekday_of_month(2026, 2, calendar.SATURDAY, 1)
    third_sat = nth_weekday_of_month(2026, 2, calendar.SATURDAY, 3)

    # Add a top-level critical node for the overall plan
    top = evaluator.add_parallel(
        id="Holiday_DIY_Project_Planning",
        desc="Answer all parts: (1) select a qualifying free kids DIY workshop in Feb 2026 and give the date; (2) select a store for purchasing supplies after 5:30 PM on Dec 24, 2025 and give closing time; (3) list four essential wreath-supply categories (general types, not brands).",
        parent=root,
        critical=True
    )

    # Record ground truth/context info used for judgment
    evaluator.add_ground_truth(
        {
            "workshop_rules": {"home depot": "first Saturday", "lowe's": "third Saturday"},
            "feb_2026_first_saturday": first_sat.isoformat() if first_sat else None,
            "feb_2026_third_saturday": third_sat.isoformat() if third_sat else None,
            "christmas_eve_2025_closing_times": {"Home Depot": "5:00 PM", "Lowe's": "6:00 PM", "Hobby Lobby": "5:30 PM"},
            "after_530_threshold": "5:30 PM"
        },
        gt_type="ground_truth"
    )

    # Run extractions (can be parallelized)
    workshop_task = evaluator.extract(
        prompt=prompt_extract_workshop(),
        template_class=WorkshopExtraction,
        extraction_name="workshop_selection"
    )
    shopping_task = evaluator.extract(
        prompt=prompt_extract_shopping(),
        template_class=ShoppingExtraction,
        extraction_name="shopping_selection"
    )
    supplies_task = evaluator.extract(
        prompt=prompt_extract_supplies(),
        template_class=SuppliesExtraction,
        extraction_name="supplies_list"
    )

    workshop, shopping, supplies = await asyncio.gather(workshop_task, shopping_task, supplies_task)

    # Build verification subtrees
    await build_workshop_selection(evaluator, top, workshop)
    await build_christmas_eve_selection(evaluator, top, shopping)
    await build_essential_supplies(evaluator, top, supplies)

    return evaluator.get_summary()