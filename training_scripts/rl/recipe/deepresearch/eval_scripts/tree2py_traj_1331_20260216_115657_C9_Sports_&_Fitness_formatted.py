import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fall2024_wmm_bq2026"
TASK_DESCRIPTION = (
    "Identify 3 World Marathon Major races from fall 2024 (September–November 2024) "
    "that fall within the Boston Marathon 2026 qualifying window (September 1, 2024 to September 12, 2025). "
    "For each race, provide: "
    "(1) Race Identity: Official full title including sponsor name, common short name, host city and country, and confirmation of World Marathon Major status; "
    "(2) Date Information: Exact race date (Month Day, Year), confirmation the race occurred in the specified month (September, October, or November 2024), and verification the date falls within the Boston 2026 qualifying window; "
    "(3) Men's Elite Results: Winner's full name and nationality, winning time in H:MM:SS format, and confirmation the time is faster than the Boston Marathon qualifying standard for men aged 18-34 (2:55:00); "
    "(4) Women's Elite Results: Winner's full name and nationality, winning time in H:MM:SS format, and confirmation the time is faster than the Boston Marathon qualifying standard for women aged 18-34 (3:25:00); "
    "(5) URL references verifying all information."
)

# Boston 2026 qualifying window (inclusive)
BQ_WINDOW_START = datetime(2024, 9, 1)
BQ_WINDOW_END = datetime(2025, 9, 12)

MEN_BQ_SECONDS = 2 * 3600 + 55 * 60  # 2:55:00
WOMEN_BQ_SECONDS = 3 * 3600 + 25 * 60  # 3:25:00

REQUIRED_MONTHS = ["September", "October", "November"]  # 2024


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RaceInfo(BaseModel):
    # Identity
    full_title: Optional[str] = None
    common_name: Optional[str] = None
    name_sources: List[str] = Field(default_factory=list)

    city: Optional[str] = None
    country: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)

    wmm_status_statement: Optional[str] = None  # e.g., "Abbott World Marathon Major"
    wmm_sources: List[str] = Field(default_factory=list)

    # Date
    date_full: Optional[str] = None  # "Month Day, Year"
    date_sources: List[str] = Field(default_factory=list)

    # Men's results
    men_winner_name: Optional[str] = None
    men_winner_nation: Optional[str] = None
    men_winner_sources: List[str] = Field(default_factory=list)
    men_win_time: Optional[str] = None
    men_time_sources: List[str] = Field(default_factory=list)

    # Women's results
    women_winner_name: Optional[str] = None
    women_winner_nation: Optional[str] = None
    women_winner_sources: List[str] = Field(default_factory=list)
    women_win_time: Optional[str] = None
    women_time_sources: List[str] = Field(default_factory=list)


class RacesExtraction(BaseModel):
    races: List[RaceInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_races() -> str:
    return """
Extract up to 5 marathon races from the answer that are presented as Abbott World Marathon Majors in fall 2024. For each race, extract the following fields exactly as provided in the answer:

For each race (object in the 'races' array), extract:
- full_title: The complete official race title including sponsor(s) if shown (e.g., "BMW BERLIN-MARATHON 2024", "Bank of America Chicago Marathon 2024"), or as written in the answer. If no sponsor is shown, use the official title presented.
- common_name: The commonly used short name (e.g., "Berlin Marathon", "Chicago Marathon").
- name_sources: An array of URL(s) that verify the race name/title; extract only URLs explicitly present in the answer.

- city: The host city of the marathon exactly as written (e.g., "Berlin").
- country: The host country (e.g., "Germany").
- location_sources: An array of URL(s) that verify the location; extract only URLs explicitly present in the answer.

- wmm_status_statement: The statement indicating Abbott World Marathon Major status, or simply "Abbott World Marathon Major" if it’s directly stated in the answer.
- wmm_sources: An array of URL(s) that verify the Abbott World Marathon Major status; extract only URLs explicitly present in the answer.

- date_full: The exact date of the 2024 race edition in the format "Month Day, Year" if shown (e.g., "September 29, 2024"). If the answer uses a different but equivalent format, extract it as-is.
- date_sources: An array of URL(s) that verify the race date; extract only URLs explicitly present in the answer.

- men_winner_name: The men's elite winner's full name.
- men_winner_nation: The men's elite winner's nationality (country name or IOC code, e.g., "ETH" or "Ethiopia").
- men_winner_sources: An array of URL(s) to verify men's winner identity/nationality; extract only URLs explicitly present in the answer.
- men_win_time: The men's winning time in H:MM:SS format if shown (or as-is if slightly different).
- men_time_sources: An array of URL(s) that verify the men's winning time; extract only URLs explicitly present in the answer.

- women_winner_name: The women's elite winner's full name.
- women_winner_nation: The women's elite winner's nationality (country name or IOC code).
- women_winner_sources: An array of URL(s) to verify women's winner identity/nationality; extract only URLs explicitly present in the answer.
- women_win_time: The women's winning time in H:MM:SS format if shown (or as-is if slightly different).
- women_time_sources: An array of URL(s) that verify the women's winning time; extract only URLs explicitly present in the answer.

IMPORTANT:
- Do NOT invent or infer URLs; only include URLs that are explicitly present in the answer (plain URLs or markdown links).
- If a field is missing in the answer for a race, set it to null (for strings) or an empty array (for lists).
- Return a JSON object with one key: "races", an array of race objects as defined above.
    """


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
MONTH_ALIASES = {
    "sep": "September",
    "sept": "September",
    "september": "September",
    "oct": "October",
    "october": "October",
    "nov": "November",
    "november": "November",
}


def extract_month_name(date_text: Optional[str]) -> Optional[str]:
    if not date_text:
        return None
    s = date_text.lower()
    for key, val in MONTH_ALIASES.items():
        if key in s:
            return val
    return None


def parse_time_to_seconds(time_str: Optional[str]) -> Optional[int]:
    if not time_str:
        return None
    s = time_str.strip().lower()
    # Remove content in parentheses and non-time annotations
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace(" ", "")
    # Normalize "h", "m", "s" to colons
    s = s.replace("hours", "h").replace("hour", "h")
    s = s.replace("minutes", "m").replace("minute", "m")
    s = s.replace("seconds", "s").replace("second", "s")
    s = s.replace("h", ":").replace("m", ":").replace("s", "")
    # Keep only digits and colons
    s = re.sub(r"[^0-9:]", "", s)

    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 3600 + m * 60 + sec
        elif len(parts) == 2:
            # If format is MM:SS, treat as minutes:seconds (assume 0 hours)
            m, sec = int(parts[0]), int(parts[1])
            return m * 60 + sec
        elif len(parts) == 1 and parts[0].isdigit():
            # Seconds only (unlikely for marathon, but handle)
            return int(parts[0])
    except Exception:
        return None
    return None


def urls_nonempty(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    for u in urls:
        if isinstance(u, str) and u.strip():
            return True
    return False


def assign_races_by_month(races: List[RaceInfo]) -> Tuple[List[RaceInfo], Dict[str, Any]]:
    """
    Try to assign races to REQUIRED_MONTHS (September, October, November)
    based on their extracted date_full. If not enough matches, fill with first remaining races
    or placeholders.
    """
    assigned: List[Optional[RaceInfo]] = [None, None, None]
    used_idx = set()
    mapping_info = {"assignments": []}

    # First pass: direct month matches
    for month_idx, month_name in enumerate(REQUIRED_MONTHS):
        for i, r in enumerate(races):
            if i in used_idx:
                continue
            m = extract_month_name(r.date_full)
            if m == month_name:
                assigned[month_idx] = r
                used_idx.add(i)
                mapping_info["assignments"].append({"target_month": month_name, "source_index": i, "strategy": "month_match"})
                break

    # Second pass: fill gaps with remaining races in original order
    remaining = [r for i, r in enumerate(races) if i not in used_idx]
    for month_idx in range(3):
        if assigned[month_idx] is None:
            if remaining:
                r = remaining.pop(0)
                assigned[month_idx] = r
                mapping_info["assignments"].append({"target_month": REQUIRED_MONTHS[month_idx], "source_index": "fallback", "strategy": "fallback_fill"})
            else:
                assigned[month_idx] = RaceInfo()
                mapping_info["assignments"].append({"target_month": REQUIRED_MONTHS[month_idx], "source_index": "placeholder", "strategy": "placeholder"})

    return [assigned[0], assigned[1], assigned[2]], mapping_info


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_list(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


async def verify_race(
    evaluator: Evaluator,
    race_parent,
    race: RaceInfo,
    race_idx: int,
    required_month: str,
) -> None:
    """
    Build the verification subtree for one race and run verifications.
    The structure mirrors the rubric with critical gating via source-availability checks.
    """
    prefix = f"Race_{race_idx + 1}"

    claims_batch: List[Tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # Identity (critical)
    identity_node = evaluator.add_parallel(
        id=f"{prefix}_Identity",
        desc="Identify the marathon race and its official status",
        parent=race_parent,
        critical=True,
    )

    # Name group (critical)
    name_node = evaluator.add_parallel(
        id=f"{prefix}_Name",
        desc="Provide the official name of the marathon",
        parent=identity_node,
        critical=True,
    )

    # Source existence for name
    evaluator.add_custom_node(
        result=urls_nonempty(race.name_sources),
        id=f"{prefix}_Name_Source",
        desc="Provide URL reference for race name verification",
        parent=name_node,
        critical=True,
    )

    # Full title
    full_title_leaf = evaluator.add_leaf(
        id=f"{prefix}_Full_Title",
        desc="State the complete official race title including sponsors",
        parent=name_node,
        critical=True,
    )
    full_title_claim = f"The official race title is '{race.full_title or ''}'."
    claims_batch.append((
        full_title_claim,
        _safe_list(race.name_sources),
        full_title_leaf,
        "Allow minor variations in sponsor naming, punctuation, and letter case. The page should clearly refer to the same marathon event."
    ))

    # Common name
    common_name_leaf = evaluator.add_leaf(
        id=f"{prefix}_Common_Name",
        desc="Provide the commonly used short name",
        parent=name_node,
        critical=True,
    )
    common_name_claim = f"The commonly used short name for the race is '{race.common_name or ''}'."
    claims_batch.append((
        common_name_claim,
        _safe_list(race.name_sources),
        common_name_leaf,
        "Accept reasonable variants (e.g., 'Berlin Marathon' vs. 'BMW Berlin Marathon') that clearly refer to the same event."
    ))

    # Location group (critical)
    location_node = evaluator.add_parallel(
        id=f"{prefix}_Location",
        desc="Specify the race location",
        parent=identity_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=urls_nonempty(race.location_sources),
        id=f"{prefix}_Location_Source",
        desc="Provide URL reference for location verification",
        parent=location_node,
        critical=True,
    )

    city_leaf = evaluator.add_leaf(
        id=f"{prefix}_City",
        desc="State the host city",
        parent=location_node,
        critical=True,
    )
    city_claim = f"The host city of the race is '{race.city or ''}'."
    claims_batch.append((
        city_claim,
        _safe_list(race.location_sources),
        city_leaf,
        "Verify the city for the 2024 edition of the race."
    ))

    country_leaf = evaluator.add_leaf(
        id=f"{prefix}_Country",
        desc="State the host country",
        parent=location_node,
        critical=True,
    )
    country_claim = f"The host country of the race is '{race.country or ''}'."
    claims_batch.append((
        country_claim,
        _safe_list(race.location_sources),
        country_leaf,
        "Verify the country for the 2024 edition of the race."
    ))

    # WMM status (critical)
    wmm_node = evaluator.add_parallel(
        id=f"{prefix}_WMM_Status",
        desc="Verify World Marathon Major designation",
        parent=identity_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=urls_nonempty(race.wmm_sources),
        id=f"{prefix}_WMM_Source",
        desc="Provide URL reference for WMM status verification",
        parent=wmm_node,
        critical=True,
    )

    wmm_leaf = evaluator.add_leaf(
        id=f"{prefix}_WMM_Confirmation",
        desc="Confirm the race is an Abbott World Marathon Major",
        parent=wmm_node,
        critical=True,
    )
    wmm_claim = "This race is part of the Abbott World Marathon Majors."
    claims_batch.append((
        wmm_claim,
        _safe_list(race.wmm_sources),
        wmm_leaf,
        "Verify that the page confirms the event is an Abbott World Marathon Major."
    ))

    # Date verification (critical)
    date_node = evaluator.add_parallel(
        id=f"{prefix}_Date_Verification",
        desc="Verify the race date and qualifying window compliance",
        parent=race_parent,
        critical=True,
    )

    # Exact date group (critical)
    exact_date_node = evaluator.add_parallel(
        id=f"{prefix}_Exact_Date",
        desc="Provide the complete race date",
        parent=date_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=urls_nonempty(race.date_sources),
        id=f"{prefix}_Date_Source",
        desc="Provide URL reference for date verification",
        parent=exact_date_node,
        critical=True,
    )

    date_full_leaf = evaluator.add_leaf(
        id=f"{prefix}_Date_Full",
        desc="State the date in format: Month Day, Year",
        parent=exact_date_node,
        critical=True,
    )
    date_full_claim = f"The race took place on '{(race.date_full or '').strip()}'."
    claims_batch.append((
        date_full_claim,
        _safe_list(race.date_sources),
        date_full_leaf,
        "Verify the exact date for the 2024 edition. Allow minor formatting differences."
    ))

    month_verify_leaf = evaluator.add_leaf(
        id=f"{prefix}_Month_Verification",
        desc=f"Confirm the month is {required_month} 2024",
        parent=exact_date_node,
        critical=True,
    )
    month_verify_claim = f"The 2024 edition of the race took place in {required_month} 2024."
    claims_batch.append((
        month_verify_claim,
        _safe_list(race.date_sources),
        month_verify_leaf,
        f"Check the date shown for the 2024 edition and confirm it is in {required_month} 2024."
    ))

    # BQ window checks (critical)
    window_node = evaluator.add_parallel(
        id=f"{prefix}_BQ_Window_Check",
        desc="Verify date falls within Boston 2026 qualifying window",
        parent=date_node,
        critical=True,
    )

    window_start_leaf = evaluator.add_leaf(
        id=f"{prefix}_Window_Start_Check",
        desc="Confirm date is on or after September 1, 2024",
        parent=window_node,
        critical=True,
    )
    window_start_claim = "The race date is on or after September 1, 2024."
    claims_batch.append((
        window_start_claim,
        _safe_list(race.date_sources),
        window_start_leaf,
        "Boston Marathon 2026 qualifying window starts on September 1, 2024 (inclusive). Confirm based on the date shown."
    ))

    window_end_leaf = evaluator.add_leaf(
        id=f"{prefix}_Window_End_Check",
        desc="Confirm date is on or before September 12, 2025",
        parent=window_node,
        critical=True,
    )
    window_end_claim = "The race date is on or before September 12, 2025."
    claims_batch.append((
        window_end_claim,
        _safe_list(race.date_sources),
        window_end_leaf,
        "Boston Marathon 2026 qualifying window ends on September 12, 2025 (inclusive). Confirm based on the date shown."
    ))

    # Men's elite results (critical)
    men_node = evaluator.add_parallel(
        id=f"{prefix}_Men_Elite_Results",
        desc="Provide men's elite race results",
        parent=race_parent,
        critical=True,
    )

    # Men's winner
    men_winner_node = evaluator.add_parallel(
        id=f"{prefix}_Men_Winner",
        desc="Identify the men's race winner",
        parent=men_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=urls_nonempty(race.men_winner_sources),
        id=f"{prefix}_Men_Winner_Source",
        desc="Provide URL reference for men's winner verification",
        parent=men_winner_node,
        critical=True,
    )

    men_winner_name_leaf = evaluator.add_leaf(
        id=f"{prefix}_Men_Winner_Name",
        desc="Provide winner's full name",
        parent=men_winner_node,
        critical=True,
    )
    men_winner_name_claim = f"The men's overall winner was '{race.men_winner_name or ''}'."
    claims_batch.append((
        men_winner_name_claim,
        _safe_list(race.men_winner_sources),
        men_winner_name_leaf,
        "Verify the overall men's winner for the race. Allow minor diacritic/case variations."
    ))

    men_winner_nation_leaf = evaluator.add_leaf(
        id=f"{prefix}_Men_Winner_Nation",
        desc="Provide winner's nationality",
        parent=men_winner_node,
        critical=True,
    )
    men_winner_nation_claim = f"The men's winner's nationality is '{race.men_winner_nation or ''}'."
    claims_batch.append((
        men_winner_nation_claim,
        _safe_list(race.men_winner_sources),
        men_winner_nation_leaf,
        "Nationality may be shown as a full country name or a three-letter code (e.g., KEN, ETH). Accept either form."
    ))

    # Men's winning time
    men_time_node = evaluator.add_parallel(
        id=f"{prefix}_Men_Winning_Time",
        desc="Provide the men's winning time",
        parent=men_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=urls_nonempty(race.men_time_sources),
        id=f"{prefix}_Men_Time_Source",
        desc="Provide URL reference for winning time verification",
        parent=men_time_node,
        critical=True,
    )

    men_time_value_leaf = evaluator.add_leaf(
        id=f"{prefix}_Men_Time_Value",
        desc="State the winning time in H:MM:SS format",
        parent=men_time_node,
        critical=True,
    )
    men_time_value_claim = f"The men's winning time was '{race.men_win_time or ''}'."
    claims_batch.append((
        men_time_value_claim,
        _safe_list(race.men_time_sources),
        men_time_value_leaf,
        "Verify the overall men's winning time for the race. Allow trivial formatting differences and rounding within 1 second."
    ))

    # Men's BQ comparison (custom logic check)
    men_seconds = parse_time_to_seconds(race.men_win_time)
    evaluator.add_custom_node(
        result=(men_seconds is not None and men_seconds < MEN_BQ_SECONDS),
        id=f"{prefix}_Men_BQ_Comparison",
        desc="Confirm time meets Boston qualifying standards (faster than 2:55:00 for men 18-34)",
        parent=men_time_node,
        critical=True,
    )

    # Women's elite results (critical)
    women_node = evaluator.add_parallel(
        id=f"{prefix}_Women_Elite_Results",
        desc="Provide women's elite race results",
        parent=race_parent,
        critical=True,
    )

    # Women's winner
    women_winner_node = evaluator.add_parallel(
        id=f"{prefix}_Women_Winner",
        desc="Identify the women's race winner",
        parent=women_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=urls_nonempty(race.women_winner_sources),
        id=f"{prefix}_Women_Winner_Source",
        desc="Provide URL reference for women's winner verification",
        parent=women_winner_node,
        critical=True,
    )

    women_winner_name_leaf = evaluator.add_leaf(
        id=f"{prefix}_Women_Winner_Name",
        desc="Provide winner's full name",
        parent=women_winner_node,
        critical=True,
    )
    women_winner_name_claim = f"The women's overall winner was '{race.women_winner_name or ''}'."
    claims_batch.append((
        women_winner_name_claim,
        _safe_list(race.women_winner_sources),
        women_winner_name_leaf,
        "Verify the overall women's winner for the race. Allow minor diacritic/case variations."
    ))

    women_winner_nation_leaf = evaluator.add_leaf(
        id=f"{prefix}_Women_Winner_Nation",
        desc="Provide winner's nationality",
        parent=women_winner_node,
        critical=True,
    )
    women_winner_nation_claim = f"The women's winner's nationality is '{race.women_winner_nation or ''}'."
    claims_batch.append((
        women_winner_nation_claim,
        _safe_list(race.women_winner_sources),
        women_winner_nation_leaf,
        "Nationality may be shown as a full country name or a three-letter code. Accept either form."
    ))

    # Women's winning time
    women_time_node = evaluator.add_parallel(
        id=f"{prefix}_Women_Winning_Time",
        desc="Provide the women's winning time",
        parent=women_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=urls_nonempty(race.women_time_sources),
        id=f"{prefix}_Women_Time_Source",
        desc="Provide URL reference for winning time verification",
        parent=women_time_node,
        critical=True,
    )

    women_time_value_leaf = evaluator.add_leaf(
        id=f"{prefix}_Women_Time_Value",
        desc="State the winning time in H:MM:SS format",
        parent=women_time_node,
        critical=True,
    )
    women_time_value_claim = f"The women's winning time was '{race.women_win_time or ''}'."
    claims_batch.append((
        women_time_value_claim,
        _safe_list(race.women_time_sources),
        women_time_value_leaf,
        "Verify the overall women's winning time for the race. Allow trivial formatting differences and rounding within 1 second."
    ))

    # Women's BQ comparison (custom logic check)
    women_seconds = parse_time_to_seconds(race.women_win_time)
    evaluator.add_custom_node(
        result=(women_seconds is not None and women_seconds < WOMEN_BQ_SECONDS),
        id=f"{prefix}_Women_BQ_Comparison",
        desc="Confirm time meets Boston qualifying standards (faster than 3:25:00 for women 18-34)",
        parent=women_time_node,
        critical=True,
    )

    # Execute all verifications in batch
    if claims_batch:
        await evaluator.batch_verify(claims_batch)


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the Fall 2024 World Marathon Majors within Boston 2026 qualifying window task.
    """
    evaluator = Evaluator()

    # Root node: parallel aggregation across three races; allow partial credit (non-critical root)
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

    # Extract structured race info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_races(),
        template_class=RacesExtraction,
        extraction_name="races_extraction",
    )

    # Assign races to required months (September, October, November)
    selected_races, mapping_info = assign_races_by_month(extracted.races[:5] if extracted.races else [])
    evaluator.add_custom_info(
        info={
            "required_months": REQUIRED_MONTHS,
            "bq_window_start": BQ_WINDOW_START.strftime("%Y-%m-%d"),
            "bq_window_end": BQ_WINDOW_END.strftime("%Y-%m-%d"),
            "men_bq_threshold": "2:55:00",
            "women_bq_threshold": "3:25:00",
            "assignment": mapping_info,
            "extracted_count": len(extracted.races) if extracted and extracted.races else 0,
        },
        info_type="assignment_info",
    )

    # Build three top-level race nodes (non-critical to allow partial scoring)
    race_nodes = []
    for i, month in enumerate(REQUIRED_MONTHS):
        race_node = evaluator.add_parallel(
            id=f"Race_{i + 1}",
            desc=(
                f"{['First','Second','Third'][i]} race: A World Marathon Major held in {month} 2024"
            ),
            parent=root,
            critical=False,
        )
        race_nodes.append(race_node)

    # Verify each race subtree
    for i in range(3):
        race = selected_races[i] if i < len(selected_races) else RaceInfo()
        await verify_race(
            evaluator=evaluator,
            race_parent=race_nodes[i],
            race=race,
            race_idx=i,
            required_month=REQUIRED_MONTHS[i],
        )

    return evaluator.get_summary()