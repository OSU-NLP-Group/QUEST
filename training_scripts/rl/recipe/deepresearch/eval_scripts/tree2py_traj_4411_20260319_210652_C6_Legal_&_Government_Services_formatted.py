import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "multi_state_short_session_2026"
TASK_DESCRIPTION = (
    "I am a government relations professional planning a coordinated advocacy campaign across multiple states for "
    "legislation that needs to be introduced early in the 2026 legislative session. I need to identify four states "
    "where I can work efficiently within shorter session windows while taking advantage of procedural flexibility.\n\n"
    "Please identify exactly four U.S. states that meet ALL of the following criteria for their 2026 regular legislative sessions:\n\n"
    "1. The regular session must last 90 calendar days or fewer (from start date to adjournment date)\n"
    "2. The state must allow bill introductions for at least 10 days after the session begins (the bill introduction deadline cannot be on the first day of session)\n"
    "3. The state must allow legislation to carry over from the 2025 session to the 2026 session\n"
    "4. The 2026 regular session must begin between January 5, 2026 and January 31, 2026\n"
    "5. The state must have an established crossover deadline (the date by which bills must pass from one chamber to the other)\n"
    "6. The state must hold a regular legislative session in 2026 (not states that skip even-numbered years)\n\n"
    "Additionally, the four states you identify must represent at least three different U.S. Census regions (Northeast, South, Midwest, West) to ensure geographic diversity in my campaign.\n\n"
    "For each state, provide:\n"
    "- State name\n"
    "- 2026 session start date\n"
    "- 2026 session adjournment date\n"
    "- Session duration in calendar days\n"
    "- Bill introduction deadline\n"
    "- Crossover deadline\n"
    "- Confirmation that the state allows carryover from 2025 to 2026\n"
    "- U.S. Census region\n"
    "- A link to the official source verifying this information (state legislative website, NCSL, MultiState, or similar authoritative resource)"
)

# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class StateEntry(BaseModel):
    state_name: Optional[str] = None

    # Dates as strings exactly as provided in the answer
    session_start_date: Optional[str] = None
    session_adjourn_date: Optional[str] = None
    session_duration_days: Optional[str] = None

    introduction_deadline_date: Optional[str] = None
    crossover_deadline: Optional[str] = None

    carryover_note: Optional[str] = None
    regular_session_2026_note: Optional[str] = None

    census_region: Optional[str] = None

    # Source URLs (authoritative links) – per-aspect if cited; otherwise use general
    sources_session_dates: List[str] = Field(default_factory=list)
    sources_introduction: List[str] = Field(default_factory=list)
    sources_carryover: List[str] = Field(default_factory=list)
    sources_crossover: List[str] = Field(default_factory=list)
    sources_general: List[str] = Field(default_factory=list)


class CampaignStatesExtraction(BaseModel):
    states: List[StateEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_states() -> str:
    return """
    Extract exactly the first four U.S. states proposed in the answer that are intended to meet the specified 2026 legislative session criteria.
    For each of the four states (in the order they appear), extract the following fields as strings exactly as stated in the answer (do not normalize or reformat text), and extract URLs that the answer explicitly provides:

    Required fields for each state object:
    - state_name
    - session_start_date
    - session_adjourn_date
    - session_duration_days  (if a duration is explicitly provided in the answer; otherwise null)
    - introduction_deadline_date  (e.g., last day to introduce bills; if not provided, set to null)
    - crossover_deadline  (if a specific date or description is provided; if not, set to null)
    - carryover_note  (any statement indicating carryover from 2025 to 2026; if not provided, null)
    - regular_session_2026_note  (any statement confirming a regular session occurs in 2026; if not provided, null)
    - census_region  (Northeast, South, Midwest, or West; if not provided, set to null)

    Source URLs:
    - sources_session_dates: URLs that support the 2026 session start/adjournment dates.
    - sources_introduction: URLs that support the bill introduction deadline timing.
    - sources_carryover: URLs that support carryover from 2025 to 2026 (two-year session or similar).
    - sources_crossover: URLs that support the crossover deadline.
    - sources_general: Any other authoritative URLs provided for this state.
    
    SPECIAL RULES:
    - Only include URLs explicitly present in the answer (plain URLs or URLs inside markdown links).
    - Do not invent or infer URLs. If no URL is given for a field, leave that field’s URL list empty.
    - Keep date strings exactly as shown in the answer (e.g., “January 9, 2026” or “1/9/2026”).
    - If the answer lists more than four states, only take the first four. If it lists fewer than four, include as many as are present.

    Return a JSON object with a top-level "states" array of length up to 4 where each element is a state object with the fields specified above.
    """


# --------------------------------------------------------------------------- #
# Helpers: dates, sources, regions                                            #
# --------------------------------------------------------------------------- #
def _clean_date_text(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    # Normalize month abbreviations like "Jan." -> "Jan"
    s = s.replace("Jan.", "Jan").replace("Feb.", "Feb").replace("Mar.", "Mar").replace("Apr.", "Apr") \
         .replace("Jun.", "Jun").replace("Jul.", "Jul").replace("Aug.", "Aug").replace("Sep.", "Sep") \
         .replace("Sept.", "Sep").replace("Oct.", "Oct").replace("Nov.", "Nov").replace("Dec.", "Dec")
    # Remove ordinal suffixes: 1st, 2nd, 3rd, 4th, etc.
    for suf in ["st", "nd", "rd", "th"]:
        s = s.replace(f" {suf},", ",").replace(f"{suf},", ",").replace(f" {suf} ", " ")
    # Remove unicode dashes
    s = s.replace("–", "-").replace("—", "-")
    return s


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    s = _clean_date_text(date_str)

    # Try various common formats
    fmts = [
        "%B %d, %Y",   # January 5, 2026
        "%b %d, %Y",   # Jan 5, 2026
        "%B %d %Y",    # January 5 2026
        "%b %d %Y",    # Jan 5 2026
        "%m/%d/%Y",    # 01/05/2026
        "%m-%d-%Y",    # 01-05-2026
        "%Y-%m-%d",    # 2026-01-05
        "%d %B %Y",    # 5 January 2026
        "%d %b %Y",    # 5 Jan 2026
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue

    # Try to extract a plain month name and day, year indirectly (very loose)
    # If fails, return None
    return None


def inclusive_days(start: Optional[datetime], end: Optional[datetime]) -> Optional[int]:
    if not start or not end:
        return None
    delta = (end - start).days
    return delta + 1 if delta >= 0 else None


def is_within_range(target: Optional[datetime], start: datetime, end: datetime) -> bool:
    if not target:
        return False
    return start <= target <= end


def dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls or []:
        if not isinstance(u, str):
            continue
        su = u.strip()
        if not su:
            continue
        # Only accept plausible URLs
        if not (su.startswith("http://") or su.startswith("https://")):
            continue
        if su not in seen:
            seen.add(su)
            out.append(su)
    return out


def gather_sources(entry: StateEntry, prefer: str) -> List[str]:
    """
    prefer: one of 'dates', 'intro', 'carryover', 'crossover', 'general'
    Returns the best-available list (preferred list if non-empty; otherwise falls back to general).
    """
    mapping = {
        "dates": entry.sources_session_dates,
        "intro": entry.sources_introduction,
        "carryover": entry.sources_carryover,
        "crossover": entry.sources_crossover,
        "general": entry.sources_general,
    }
    preferred = dedupe_urls(mapping.get(prefer, []))
    if preferred:
        return preferred
    # Fall back order: general, then any others
    fallback = dedupe_urls(entry.sources_general)
    if fallback:
        return fallback
    # Try any other list that is non-empty
    for k, v in mapping.items():
        if k == prefer or k == "general":
            continue
        vv = dedupe_urls(v)
        if vv:
            return vv
    return []


def normalize_region_name(region: Optional[str]) -> Optional[str]:
    if not region:
        return None
    r = region.strip().lower()
    # Normalize common variants/abbreviations
    mapping = {
        "northeast": "Northeast",
        "north east": "Northeast",
        "ne": "Northeast",
        "midwest": "Midwest",
        "mid-west": "Midwest",
        "mw": "Midwest",
        "south": "South",
        "s": "South",
        "west": "West",
        "w": "West",
        "pacific": "West",
        "mountain": "West",
        "east north central": "Midwest",
        "west north central": "Midwest",
        "south atlantic": "South",
        "east south central": "South",
        "west south central": "South",
        "new england": "Northeast",
        "middle atlantic": "Northeast"
    }
    # Try direct mapping
    if r in mapping:
        return mapping[r]
    # Try partial contains
    for key, val in mapping.items():
        if key in r:
            return val
    # Title-case fallback
    return region.strip().title()


# --------------------------------------------------------------------------- #
# Verification utilities                                                      #
# --------------------------------------------------------------------------- #
async def verify_with_sources_or_fail(
    evaluator: Evaluator,
    *,
    id: str,
    desc: str,
    parent,
    critical: bool,
    claim: str,
    sources: List[str],
    additional_instruction: str
):
    """
    Add a leaf node and verify with URLs. If no sources provided, mark the node failed immediately
    to enforce source-grounding for 'source' leaves.
    """
    if sources:
        node = evaluator.add_leaf(id=id, desc=desc, parent=parent, critical=critical)
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=sources,
            additional_instruction=additional_instruction,
        )
    else:
        # No sources – fail this source-grounded check by design
        evaluator.add_leaf(
            id=id,
            desc=f"{desc} (failed: no source URLs provided in the answer)",
            parent=parent,
            critical=critical,
            score=0.0,
            status="failed"
        )


# --------------------------------------------------------------------------- #
# Per-state verification                                                      #
# --------------------------------------------------------------------------- #
async def verify_one_state(evaluator: Evaluator, parent_node, entry: StateEntry, idx: int):
    """
    Build the verification subtree for a single state according to the rubric.
    """
    sid = idx + 1
    state_label = entry.state_name or f"State #{sid}"

    state_node = evaluator.add_parallel(
        id=f"S{sid}",
        desc=f"{state_label} – meets all legislative session requirements",
        parent=parent_node,
        critical=False
    )

    # ----------------------------- 1) Session Duration (sequential) -----------------------------
    dur_node = evaluator.add_sequential(
        id=f"S{sid}_Session_Duration",
        desc="Verify the 2026 regular session duration is 90 calendar days or fewer",
        parent=state_node,
        critical=True
    )

    # 1.1 Session dates identified (existence check)
    has_dates = bool(entry.session_start_date and entry.session_adjourn_date)
    evaluator.add_custom_node(
        result=has_dates,
        id=f"S{sid}_Session_Dates",
        desc="The session start and adjournment dates are identified",
        parent=dur_node,
        critical=True
    )

    # 1.2 Duration calculation <= 90
    start_dt = parse_date(entry.session_start_date)
    end_dt = parse_date(entry.session_adjourn_date)
    dur_days = inclusive_days(start_dt, end_dt)
    duration_ok = (dur_days is not None) and (dur_days <= 90)
    evaluator.add_custom_node(
        result=bool(duration_ok),
        id=f"S{sid}_Duration_Calculation",
        desc="The calculated duration from start to adjournment is 90 days or fewer",
        parent=dur_node,
        critical=True
    )

    # 1.3 Session dates verified by authoritative source(s)
    dates_sources = gather_sources(entry, "dates")
    await verify_with_sources_or_fail(
        evaluator,
        id=f"S{sid}_Duration_Source",
        desc="Session dates are verified through official/authoritative resource",
        parent=dur_node,
        critical=True,
        claim=f"According to the cited source(s), {state_label}'s 2026 regular session begins on {entry.session_start_date} and adjourns on {entry.session_adjourn_date}.",
        sources=dates_sources,
        additional_instruction="Accept only if the page explicitly lists 2026 regular session start and adjournment (sine die) dates for this state. Prefer state legislative calendars or authoritative aggregators (NCSL, MultiState, FiscalNote)."
    )

    # ----------------------------- 2) Bill Introduction (sequential) -----------------------------
    intro_node = evaluator.add_sequential(
        id=f"S{sid}_Bill_Introduction",
        desc="Verify bill introduction timeline requirements",
        parent=state_node,
        critical=True
    )

    # 2.1 Introduction deadline identified
    has_intro_deadline = bool(entry.introduction_deadline_date)
    evaluator.add_custom_node(
        result=has_intro_deadline,
        id=f"S{sid}_Introduction_Deadline",
        desc="The bill introduction deadline is identified",
        parent=intro_node,
        critical=True
    )

    # 2.2 Timeline calculation: at least 10 days after session start; not first day
    intro_dt = parse_date(entry.introduction_deadline_date) if has_intro_deadline else None
    at_least_10_days = False
    if start_dt and intro_dt:
        diff_days = (intro_dt - start_dt).days
        at_least_10_days = diff_days >= 10
    evaluator.add_custom_node(
        result=bool(at_least_10_days),
        id=f"S{sid}_Timeline_Calculation",
        desc="The introduction deadline allows at least 10 days after session starts (not first day of session)",
        parent=intro_node,
        critical=True
    )

    # 2.3 Introduction deadline verified by source(s)
    intro_sources = gather_sources(entry, "intro")
    await verify_with_sources_or_fail(
        evaluator,
        id=f"S{sid}_Introduction_Source",
        desc="Introduction deadline is verified through official legislative procedures documentation",
        parent=intro_node,
        critical=True,
        claim=f"According to the cited source(s), the bill introduction deadline for {state_label}'s 2026 regular session is {entry.introduction_deadline_date}.",
        sources=intro_sources,
        additional_instruction="Confirm the page states the last day to introduce new bills (or equivalent). It must be at least 10 days after the session start, and not the first day."
    )

    # ----------------------------- 3) Carryover Provision (parallel) -----------------------------
    carry_node = evaluator.add_parallel(
        id=f"S{sid}_Carryover_Provision",
        desc="Verify carryover provision exists",
        parent=state_node,
        critical=True
    )

    carry_sources = gather_sources(entry, "carryover")

    # 3.1 Carryover status (verified)
    await verify_with_sources_or_fail(
        evaluator,
        id=f"S{sid}_Carryover_Status",
        desc="The state allows legislation to carry over from 2025 to 2026",
        parent=carry_node,
        critical=True,
        claim=f"The state of {state_label} allows carryover of bills from the 2025 session into the 2026 session (two-year session or equivalent carryover policy).",
        sources=carry_sources,
        additional_instruction="Accept only if the authority (rules/legislative resources) explicitly state that bills not enacted in 2025 can continue or carry over into 2026."
    )

    # 3.2 Carryover source (explicit authority check)
    await verify_with_sources_or_fail(
        evaluator,
        id=f"S{sid}_Carryover_Source",
        desc="Carryover provision is verified through official legislative rules or an authoritative source",
        parent=carry_node,
        critical=True,
        claim=f"The cited page is an official legislative rule or authoritative source confirming {state_label}'s 2025-to-2026 carryover policy.",
        sources=carry_sources,
        additional_instruction="Evaluate whether the linked page is an official legislative rules page or a recognized authoritative aggregator (NCSL, MultiState, FiscalNote) that clearly documents 2025-to-2026 carryover."
    )

    # ----------------------------- 4) Session Start constraints (parallel) -----------------------------
    start_node = evaluator.add_parallel(
        id=f"S{sid}_Session_Start",
        desc="Verify session start date meets requirements",
        parent=state_node,
        critical=True
    )

    # 4.1 Start date range check (Jan 5 – Jan 31, 2026 inclusive)
    range_ok = is_within_range(
        start_dt,
        datetime(2026, 1, 5),
        datetime(2026, 1, 31)
    )
    evaluator.add_custom_node(
        result=bool(range_ok),
        id=f"S{sid}_Start_Date_Range",
        desc="The 2026 session begins between January 5 and January 31, 2026",
        parent=start_node,
        critical=True
    )

    # 4.2 Regular session held in 2026 – verify via dates sources
    await verify_with_sources_or_fail(
        evaluator,
        id=f"S{sid}_Regular_Session",
        desc="The state holds a regular session in 2026 (not a biennial-skip state)",
        parent=start_node,
        critical=True,
        claim=f"{state_label} holds a regular legislative session in 2026.",
        sources=dates_sources,
        additional_instruction="Confirm that the page clearly references a 2026 regular session (not a special-only session and not a state that skips even-numbered years)."
    )

    # ----------------------------- 5) Crossover Deadline (parallel) -----------------------------
    cross_node = evaluator.add_parallel(
        id=f"S{sid}_Crossover_Deadline",
        desc="Verify crossover deadline existence",
        parent=state_node,
        critical=True
    )

    cross_sources = gather_sources(entry, "crossover")
    # 5.1 Existence (prefer a specific date if provided)
    if entry.crossover_deadline:
        cross_claim = f"{state_label} has an established crossover deadline for its 2026 session on {entry.crossover_deadline}."
    else:
        cross_claim = f"{state_label} has an established crossover deadline for bills in the 2026 session."

    await verify_with_sources_or_fail(
        evaluator,
        id=f"S{sid}_Crossover_Exists",
        desc="The state has an established crossover deadline for bills",
        parent=cross_node,
        critical=True,
        claim=cross_claim,
        sources=cross_sources,
        additional_instruction="Verify that the legislative calendar or procedures define a formal 'crossover' (house of origin) deadline in 2026."
    )

    # 5.2 Source verification
    if entry.crossover_deadline:
        cross_source_claim = f"The cited page confirms the 2026 crossover deadline for {state_label} is {entry.crossover_deadline}."
    else:
        cross_source_claim = f"The cited page confirms {state_label} has a defined 2026 crossover (house-of-origin) deadline."

    await verify_with_sources_or_fail(
        evaluator,
        id=f"S{sid}_Crossover_Source",
        desc="Crossover deadline is verified through official legislative calendar",
        parent=cross_node,
        critical=True,
        claim=cross_source_claim,
        sources=cross_sources,
        additional_instruction="Accept only if the page is an official legislative calendar or a recognized authoritative aggregator explicitly listing the 2026 crossover deadline."
    )

    # Store some computed info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "state": state_label,
            "computed_session_days": dur_days,
            "start_date_parsed": start_dt.isoformat() if start_dt else None,
            "end_date_parsed": end_dt.isoformat() if end_dt else None,
            "intro_deadline_parsed": intro_dt.isoformat() if intro_dt else None
        },
        info_type="state_computed_metrics",
        info_name=f"state_{sid}_metrics"
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
    Evaluate an answer for the 'four states short session 2026' task.
    """
    # Initialize evaluator (root is non-critical by design for flexibility)
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

    # Record GT-style context (constraints snapshot)
    evaluator.add_ground_truth({
        "constraints": {
            "max_duration_days": 90,
            "min_intro_days_after_start": 10,
            "start_window": ["2026-01-05", "2026-01-31"],
            "requires_carryover_2025_to_2026": True,
            "requires_crossover_deadline": True,
            "requires_regular_session_2026": True,
            "geographic_diversity_min_regions": 3,
            "exact_states_requested": 4
        }
    }, gt_type="task_constraints")

    # Extract structured state info from the answer
    extraction: CampaignStatesExtraction = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=CampaignStatesExtraction,
        extraction_name="proposed_states"
    )

    states_list: List[StateEntry] = extraction.states[:4] if extraction.states else []
    # Pad to 4 entries with empty placeholders if needed
    while len(states_list) < 4:
        states_list.append(StateEntry())

    # Add a container node for the task (non-critical due to framework constraints on critical parents)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Find four states with 2026 legislative sessions meeting all specified criteria",
        parent=root,
        critical=False
    )

    # Build verification subtrees for each of the four states
    for i, st in enumerate(states_list[:4]):
        await verify_one_state(evaluator, task_node, st, i)

    # Geographic diversity check: at least 3 distinct Census regions among the four
    regions = [normalize_region_name(st.census_region) for st in states_list[:4] if normalize_region_name(st.census_region)]
    unique_regions = sorted(set([r for r in regions if r]))
    geo_ok = len(unique_regions) >= 3

    evaluator.add_custom_info(
        info={"regions_found": regions, "unique_regions": unique_regions, "count_unique_regions": len(unique_regions)},
        info_type="geography",
        info_name="region_diversity_details"
    )

    evaluator.add_custom_node(
        result=geo_ok,
        id="Geographic_Diversity",
        desc="The four states represent at least three different U.S. Census regions",
        parent=task_node,
        critical=True
    )

    # Return structured summary
    return evaluator.get_summary()