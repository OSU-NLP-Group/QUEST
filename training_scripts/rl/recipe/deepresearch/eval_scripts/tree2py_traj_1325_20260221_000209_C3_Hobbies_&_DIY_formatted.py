import asyncio
import logging
import re
from datetime import datetime, date, time
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lowes_birdhouse_feb2026_atlanta"
TASK_DESCRIPTION = (
    "A parent in Atlanta, Georgia wants to take their 7-year-old child to a free kids' DIY birdhouse-building "
    "workshop at a Lowe's store during the last weekend of February 2026. The parent needs to arrive at the store "
    "at least 1 hour before the workshop begins to purchase additional craft paint supplies at the same location. "
    "Identify the appropriate workshop (including the specific date and start time), specify which Lowe's store in "
    "Atlanta, GA the parent should visit, and verify that the store's operating hours on the workshop day allow for "
    "at least 1 hour of pre-workshop shopping time. Provide the workshop registration/information page URL and the "
    "store's location/hours page URL as references."
)

# Ground truth context for the "last full weekend of February 2026"
LAST_FULL_WEEKEND_DATES = {date(2026, 2, 21), date(2026, 2, 22)}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WorkshopDetails(BaseModel):
    project_type: Optional[str] = None  # e.g., "Build a Birdhouse", "Birdhouse"
    date: Optional[str] = None          # e.g., "February 21, 2026"
    start_time: Optional[str] = None    # e.g., "10:00 AM"
    age_range: Optional[str] = None     # e.g., "Ages 4-11"
    registration_url: Optional[str] = None  # URL for workshop registration/info


class StoreDetails(BaseModel):
    store_name: Optional[str] = None
    city: Optional[str] = None          # Should be "Atlanta"
    state: Optional[str] = None         # Should be "GA" or "Georgia"
    store_url: Optional[str] = None     # Store location/hours page
    opening_time_on_workshop_day: Optional[str] = None  # e.g., "9:00 AM"


class ParticipationPlanExtraction(BaseModel):
    workshop: Optional[WorkshopDetails] = None
    store: Optional[StoreDetails] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return (
        "Extract the Lowe's kids workshop and store details provided in the answer.\n"
        "Return JSON with the following structure:\n"
        "{\n"
        '  "workshop": {\n'
        '    "project_type": string or null,          // e.g., "Build a Birdhouse" or "Birdhouse"\n'
        '    "date": string or null,                  // e.g., "February 21, 2026" (prefer full month name, day, year)\n'
        '    "start_time": string or null,            // e.g., "10:00 AM" (use 12-hour format with AM/PM)\n'
        '    "age_range": string or null,             // e.g., "Ages 4-11" or similar kids age range\n'
        '    "registration_url": string or null       // The workshop registration/info URL explicitly shown in the answer\n'
        "  },\n"
        '  "store": {\n'
        '    "store_name": string or null,            // e.g., "Lowe\'s of Atlanta - Buckhead" or similar\n'
        '    "city": string or null,                  // e.g., "Atlanta"\n'
        '    "state": string or null,                 // e.g., "GA" or "Georgia"\n'
        '    "store_url": string or null,             // The store location/hours page URL explicitly shown in the answer\n'
        '    "opening_time_on_workshop_day": string or null // e.g., "9:00 AM" (opening time for the workshop day)\n'
        "  }\n"
        "}\n"
        "Rules:\n"
        "- Extract ONLY what is explicitly present in the answer. Do not invent.\n"
        "- For any missing item, return null.\n"
        "- For URLs, extract the actual URL strings (plain or from markdown links). If protocol is missing, prepend http://.\n"
        "- Prefer standard formats similar to the examples above for date/time if available.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions for time/date parsing and logic                            #
# --------------------------------------------------------------------------- #
def _try_parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    s = date_str.strip()
    patterns = [
        "%B %d, %Y",   # February 21, 2026
        "%b %d, %Y",   # Feb 21, 2026
        "%B %d %Y",    # February 21 2026
        "%b %d %Y",    # Feb 21 2026
        "%Y-%m-%d",    # 2026-02-21
        "%m/%d/%Y",    # 02/21/2026
    ]
    for p in patterns:
        try:
            return datetime.strptime(s, p).date()
        except Exception:
            continue
    # Fallback: try to detect "Feb 21-22, 2026" or similar range mentioning
    if re.search(r"\b(Feb|February)\b", s, flags=re.IGNORECASE):
        if re.search(r"\b21\b", s) or re.search(r"\b22\b", s):
            # Cannot pick one specific day; but indicates last weekend mention
            # Return a sentinel value to indicate it's within that weekend
            # We choose the first day of the weekend for logical checks
            return date(2026, 2, 21)
    return None


def _normalize_time_string(t: str) -> str:
    s = t.strip().lower()
    s = s.replace(".", "")  # handle a.m./p.m.
    s = s.replace("am", " AM").replace("pm", " PM")
    s = re.sub(r"\s+", " ", s)
    s = s.upper()
    return s


def _try_parse_time(time_str: Optional[str]) -> Optional[time]:
    if not time_str:
        return None
    s = _normalize_time_string(time_str)
    candidates = [s, s.replace(" ", " ")]  # ensure normalization variants
    patterns = [
        "%I:%M %p",  # 10:00 AM
        "%I %p",     # 10 AM
    ]
    for cand in candidates:
        for p in patterns:
            try:
                return datetime.strptime(cand, p).time()
            except Exception:
                continue
    return None


def _is_last_full_weekend_feb_2026(d: Optional[date]) -> bool:
    if d is None:
        return False
    return d in LAST_FULL_WEEKEND_DATES


def _minutes_between(opening: Optional[time], start: Optional[time]) -> Optional[int]:
    if opening is None or start is None:
        return None
    # Compute minutes difference on the same day in local time assumption
    opening_dt = datetime(2026, 2, 21, opening.hour, opening.minute)  # date chosen only for difference calc
    start_dt = datetime(2026, 2, 21, start.hour, start.minute)
    diff = start_dt - opening_dt
    minutes = int(diff.total_seconds() // 60)
    return minutes


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_workshop_verification(evaluator: Evaluator, parent_node, plan: ParticipationPlanExtraction) -> None:
    """
    Build and verify the 'Workshop_Identification_And_Eligibility' subtree, including:
    - Birdhouse project type
    - Date within last full weekend of Feb 2026
    - Date and start time presented and supported by source
    - Age compatibility (includes 7-year-old)
    - Workshop info source page
    """
    wk = plan.workshop or WorkshopDetails()
    wk_url = wk.registration_url or ""

    # Parent sequential (critical)
    wk_main = evaluator.add_sequential(
        id="Workshop_Identification_And_Eligibility",
        desc="Identify the correct birdhouse workshop that meets all specified criteria (type, date, age-appropriateness) at a Lowe's location in Atlanta, GA",
        parent=parent_node,
        critical=True
    )

    # Details verification parallel (critical)
    wk_details = evaluator.add_parallel(
        id="Workshop_Details_Verification",
        desc="Verify that the identified workshop matches all required specifications",
        parent=wk_main,
        critical=True
    )

    # 1) Birdhouse project type (critical leaf, verify by URL)
    birdhouse_node = evaluator.add_leaf(
        id="Birdhouse_Project_Type",
        desc="The workshop project type is specifically for building a birdhouse",
        parent=wk_details,
        critical=True
    )
    birdhouse_claim = "This Lowe's kids workshop involves building a birdhouse (e.g., 'Build a Birdhouse')."
    await evaluator.verify(
        claim=birdhouse_claim,
        node=birdhouse_node,
        sources=wk_url,
        additional_instruction="Pass if the page clearly indicates the kids workshop project is a birdhouse. Minor naming variations like 'Bird House' or 'DIY birdhouse' are acceptable."
    )

    # 2) Last weekend of Feb 2026 (critical leaf, computed logic as custom node)
    d_obj = _try_parse_date(wk.date)
    last_weekend_ok = _is_last_full_weekend_feb_2026(d_obj)
    evaluator.add_custom_node(
        result=last_weekend_ok,
        id="Last_Weekend_February_2026",
        desc="The workshop is scheduled during the last full weekend of February 2026 (February 21-22, 2026)",
        parent=wk_details,
        critical=True
    )

    # 3) Workshop date and time provided and supported by source (critical leaf, verify by URL)
    dt_node = evaluator.add_leaf(
        id="Workshop_Date_And_Time",
        desc="The specific date and start time of the workshop are provided (e.g., February 21, 2026 at 10:00 AM)",
        parent=wk_details,
        critical=True
    )
    dt_claim = f"The workshop is scheduled for {wk.date or '[date missing]'} starting at {wk.start_time or '[start time missing]'}."
    await evaluator.verify(
        claim=dt_claim,
        node=dt_node,
        sources=wk_url,
        additional_instruction="Confirm the page explicitly shows both the event date and the start time for the kids workshop."
    )

    # 4) Age compatibility (critical leaf, verify by URL)
    age_node = evaluator.add_leaf(
        id="Age_Compatibility",
        desc="The workshop accommodates children ages 4-11, which includes the 7-year-old child",
        parent=wk_details,
        critical=True
    )
    age_claim = "This kids workshop is appropriate for a 7-year-old (e.g., the page states an age range that includes age 7, such as ages 4–11)."
    await evaluator.verify(
        claim=age_claim,
        node=age_node,
        sources=wk_url,
        additional_instruction="Pass if the page indicates an age range including age 7 (e.g., ages 4–11). Reasonable phrasing variants are acceptable."
    )

    # Workshop information source page (critical leaf, verify by URL)
    wk_src_node = evaluator.add_leaf(
        id="Workshop_Information_Source",
        desc="Provide the URL where the workshop details (including date, time, and registration information) can be verified",
        parent=wk_main,
        critical=True
    )
    wk_src_claim = "This page provides the workshop details, including date, start time, and registration/information."
    await evaluator.verify(
        claim=wk_src_claim,
        node=wk_src_node,
        sources=wk_url,
        additional_instruction="Pass if the page contains event details and a way to register or learn more."
    )


async def build_store_verification(evaluator: Evaluator, parent_node, plan: ParticipationPlanExtraction) -> None:
    """
    Build and verify the 'Store_Logistics_And_Shopping_Feasibility' subtree:
    - Store identification (Atlanta, GA)
    - Store hours allow pre-workshop shopping (opens before the workshop; minimum 1-hour shopping window)
    - Store information source page shows location and hours
    """
    wk = plan.workshop or WorkshopDetails()
    st = plan.store or StoreDetails()

    wk_date_obj = _try_parse_date(wk.date)
    wk_start_time_obj = _try_parse_time(wk.start_time)
    store_open_time_obj = _try_parse_time(st.opening_time_on_workshop_day)

    # Parent sequential (critical)
    store_main = evaluator.add_sequential(
        id="Store_Logistics_And_Shopping_Feasibility",
        desc="Verify that the selected Lowe's store location in Atlanta, GA can accommodate both pre-workshop shopping and workshop attendance",
        parent=parent_node,
        critical=True
    )

    # Store identification and hours (parallel, critical)
    store_id_hours = evaluator.add_parallel(
        id="Store_Identification_And_Hours",
        desc="Identify a specific Lowe's store in Atlanta, GA and verify its operating hours allow for pre-workshop shopping",
        parent=store_main,
        critical=True
    )

    # 1) Atlanta, GA location (critical leaf, verify by store URL)
    atl_loc_node = evaluator.add_leaf(
        id="Atlanta_GA_Location",
        desc="The store is located in Atlanta, Georgia",
        parent=store_id_hours,
        critical=True
    )
    atl_loc_claim = "This Lowe's store is located in Atlanta, GA."
    await evaluator.verify(
        claim=atl_loc_claim,
        node=atl_loc_node,
        sources=st.store_url or "",
        additional_instruction="Pass if the store page shows an Atlanta, Georgia address. Variations like 'Atlanta, GA' or neighborhood names within Atlanta are acceptable."
    )

    # 2) Shopping time feasibility (sequential, critical)
    shopping_seq = evaluator.add_sequential(
        id="Shopping_Time_Feasibility",
        desc="Verify that the store's opening hours on the workshop day allow for at least 1 hour of shopping before the workshop starts",
        parent=store_id_hours,
        critical=True
    )

    # 2.1) Store opens before workshop start (critical custom leaf)
    opens_before = False
    if wk_start_time_obj and store_open_time_obj:
        # Store must open earlier than the workshop start time
        opens_before = (
            (store_open_time_obj.hour, store_open_time_obj.minute) <
            (wk_start_time_obj.hour, wk_start_time_obj.minute)
        )

    evaluator.add_custom_node(
        result=bool(opens_before),
        id="Store_Opens_Before_Workshop",
        desc="The store opens before the workshop start time on the workshop day",
        parent=shopping_seq,
        critical=True
    )

    # 2.2) Minimum 1-hour shopping window (critical custom leaf)
    minutes_gap = _minutes_between(store_open_time_obj, wk_start_time_obj)
    at_least_one_hour = (minutes_gap is not None and minutes_gap >= 60)
    evaluator.add_custom_node(
        result=bool(at_least_one_hour),
        id="Minimum_One_Hour_Shopping_Window",
        desc="There is at least a 1-hour gap between the store opening time and the workshop start time, allowing for craft supply shopping",
        parent=shopping_seq,
        critical=True
    )

    # Add custom info for transparency
    evaluator.add_custom_info(
        {
            "workshop_date_extracted": wk.date,
            "workshop_start_time_extracted": wk.start_time,
            "store_opening_time_extracted": st.opening_time_on_workshop_day,
            "computed_minutes_between_open_and_workshop": minutes_gap,
            "meets_1_hour_requirement": at_least_one_hour
        },
        info_type="computed_timing",
        info_name="shopping_time_computation"
    )

    # Store information source (critical leaf, verify by URL)
    store_src_node = evaluator.add_leaf(
        id="Store_Information_Source",
        desc="Provide the URL where the store's location, operating hours, and workshop participation can be verified",
        parent=store_main,
        critical=True
    )
    store_src_claim = "This page shows the Lowe's store location and operating hours."
    await evaluator.verify(
        claim=store_src_claim,
        node=store_src_node,
        sources=st.store_url or "",
        additional_instruction="Pass if the page includes store address and hours of operation."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Lowe's birdhouse workshop plan in Atlanta during the last weekend of Feb 2026.
    """
    # Initialize evaluator with SEQUENTIAL strategy at root (represents complete plan)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Add a top-level node to mirror the rubric's named root (optional; keeps structure clear)
    plan_root = evaluator.add_sequential(
        id="Complete_Workshop_Participation_Plan",
        desc="Verify that a complete and feasible plan exists for a 7-year-old child to attend a free birdhouse-building workshop at a Lowe's store in Atlanta, GA during the last weekend of February 2026, with time allocated for pre-workshop shopping at the same store",
        parent=root,
        critical=False
    )

    # Extract structured plan details from the answer
    plan = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=ParticipationPlanExtraction,
        extraction_name="participation_plan"
    )

    # Add Ground Truth info context
    evaluator.add_ground_truth({
        "last_full_weekend_feb_2026_dates": [d.isoformat() for d in sorted(LAST_FULL_WEEKEND_DATES)],
        "requirement": "Store must open at least 60 minutes before workshop start"
    }, gt_type="constraints")

    # Build and verify workshop subtree
    await build_workshop_verification(evaluator, plan_root, plan)

    # Build and verify store subtree
    await build_store_verification(evaluator, plan_root, plan)

    # Return evaluation summary
    return evaluator.get_summary()