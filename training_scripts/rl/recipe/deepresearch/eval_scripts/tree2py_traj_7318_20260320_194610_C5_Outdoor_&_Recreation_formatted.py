import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "edinburgh_zoo_plan_march2026"
TASK_DESCRIPTION = """
A family is planning a visit to Edinburgh Zoo during their Scotland trip in March 2026. They want to make the most of their experience by seeing the zoo's most unique animals and attending a special weekend event. Specifically, they want to: (1) See the three types of rare/special animals that make Edinburgh Zoo distinctive (including the only animals of a specific koala subspecies found anywhere in the UK), (2) Attend the penguin event that occurs on weekends, (3) Optimize their visit timing for the best animal viewing experience, and (4) Plan for a lunch break using the zoo's facilities. Create a complete visit plan that includes: which day(s) of the week they should visit (with justification based on the penguin event schedule), the three specific types of unique animals they should prioritize (including what makes each special), complete details of the penguin event (name, schedule, duration), recommended arrival time and total visit duration, and appropriate on-site facilities for their lunch break.
"""

# Ground truth facts we expect to be supported by sources from the answer
GROUND_TRUTH = {
    "wee_waddle": {
        "name": "Wee Waddle",
        "days": ["Thursday", "Friday", "Saturday", "Sunday"],
        "start_time": "2:15 PM",
        "duration": "about 45 minutes"
    },
    "unique_animals": {
        "koalas": {
            "subspecies": "Queensland koalas",
            "names": ["Talara", "Myaree"],
            "only_in_uk": True
        },
        "red_pandas": {
            "location_detail": "behind Penguins Rock"
        },
        "giraffes": {
            "subspecies": "Nubian giraffes",
            "names": ["Arrow", "Gerald", "Fennessy", "Gilbert"],
            "count": 4
        }
    },
    "zoo_size_acres": 82
}


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class DaySelection(BaseModel):
    visit_days: List[str] = Field(default_factory=list)
    justification_text: Optional[str] = None


class EventDetails(BaseModel):
    event_name: Optional[str] = None
    days_active: List[str] = Field(default_factory=list)
    start_time: Optional[str] = None
    duration_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class KoalaInfo(BaseModel):
    names: List[str] = Field(default_factory=list)
    subspecies_text: Optional[str] = None
    only_in_uk_claim_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RedPandaInfo(BaseModel):
    mentioned_text: Optional[str] = None
    location_detail: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class GiraffeInfo(BaseModel):
    subspecies_text: Optional[str] = None
    names: List[str] = Field(default_factory=list)
    count_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TimingInfo(BaseModel):
    arrival_time_text: Optional[str] = None
    morning_guidance_text: Optional[str] = None
    total_duration_text: Optional[str] = None
    zoo_size_acres_text: Optional[str] = None
    zoo_size_sources: List[str] = Field(default_factory=list)


class LunchInfo(BaseModel):
    lunch_in_plan: Optional[bool] = None
    facilities_mentioned: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class VisitPlanExtraction(BaseModel):
    day_selection: Optional[DaySelection] = None
    event_details: Optional[EventDetails] = None
    koalas: Optional[KoalaInfo] = None
    red_pandas: Optional[RedPandaInfo] = None
    giraffes: Optional[GiraffeInfo] = None
    timing: Optional[TimingInfo] = None
    lunch: Optional[LunchInfo] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_visit_plan() -> str:
    return """
Extract a structured Edinburgh Zoo visit plan from the answer. Return a JSON object with these fields:

1) day_selection:
   - visit_days: array of the specific day(s) of the week recommended to visit. Use full weekday names: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday. If a range like "weekend" is given, map to ['Saturday','Sunday'].
   - justification_text: a short quote from the answer that explains why those day(s) were chosen (explicitly referencing the penguin event schedule).

2) event_details (the penguin event):
   - event_name: the event’s official name (e.g., "Wee Waddle") as written in the answer.
   - days_active: array of days the event runs, as claimed in the answer (use full weekday names).
   - start_time: the event’s start time as written (e.g., "2:15 PM", allow formats like "2.15pm").
   - duration_text: the event duration as written (e.g., "~45 minutes", "about 45 minutes").
   - sources: URLs explicitly provided in the answer that support these event details (include all relevant URLs).

3) koalas:
   - names: array of individual koala names mentioned (e.g., ["Talara","Myaree"]).
   - subspecies_text: the subspecies or wording indicating "Queensland koalas" if present.
   - only_in_uk_claim_text: text that states they are the only Queensland koalas in the UK (if present).
   - sources: URLs supporting the koala facts (include all relevant URLs from the answer).

4) red_pandas:
   - mentioned_text: short quote/phrase indicating red pandas are prioritized.
   - location_detail: the distinctive placement detail (e.g., "behind Penguins Rock") if present.
   - sources: URLs supporting the red panda info/location.

5) giraffes:
   - subspecies_text: the subspecies text (e.g., "Nubian giraffes") if present.
   - names: array of the individual giraffe names mentioned.
   - count_text: text/number indicating how many there are (e.g., "four") if present.
   - sources: URLs supporting giraffe names/count/subspecies.

6) timing:
   - arrival_time_text: the recommended arrival time (as a specific time string if present).
   - morning_guidance_text: text that recommends arriving in the morning / early for best viewing (if present).
   - total_duration_text: the total visit duration estimate text (e.g., "4 hours", "3–5 hours").
   - zoo_size_acres_text: text stating the zoo size in acres if mentioned (e.g., "82 acres").
   - zoo_size_sources: URLs supporting the zoo size fact (if provided).

7) lunch:
   - lunch_in_plan: true/false if the answer explicitly includes a lunch break in the plan.
   - facilities_mentioned: array of specific on-site lunch facilities/areas mentioned (e.g., "hilltop picnic area", "main lawn").
   - sources: URLs supporting the picnic/lunch facilities.

GENERAL RULES:
- Extract only what appears in the answer; do not invent anything. If something is missing, set it to null or [] accordingly.
- For any URL fields, only extract valid URLs mentioned in the answer (plain links or markdown). If none, return an empty list.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_day(day: str) -> Optional[str]:
    if not day:
        return None
    s = day.strip().lower()
    if s.startswith("mon"):
        return "Mon"
    if s.startswith("tue"):
        return "Tue"
    if s.startswith("wed"):
        return "Wed"
    if s.startswith("thu"):
        return "Thu"
    if s.startswith("fri"):
        return "Fri"
    if s.startswith("sat"):
        return "Sat"
    if s.startswith("sun"):
        return "Sun"
    return None


def _days_subset_of_thu_to_sun(days: List[str]) -> bool:
    if not days:
        return False
    allowed = {"Thu", "Fri", "Sat", "Sun"}
    normalized = {_normalize_day(d) for d in days if d}
    normalized.discard(None)
    return len(normalized) > 0 and normalized.issubset(allowed)


def _has_specific_time(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept forms like "9:30 AM", "09:00", "2.15pm", etc.
    patterns = [
        r"\b\d{1,2}:\d{2}\s*(am|pm|AM|PM)\b",
        r"\b\d{1,2}\.\d{2}\s*(am|pm|AM|PM)\b",
        r"\b\d{1,2}\s*(am|pm|AM|PM)\b",
        r"\b\d{1,2}:\d{2}\b",
    ]
    return any(re.search(p, text) for p in patterns)


def _list_contains_any(strings: List[str], keywords: List[str]) -> bool:
    s_join = " | ".join(strings).lower()
    return any(kw.lower() in s_join for kw in keywords)


# --------------------------------------------------------------------------- #
# Verification Subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_penguin_event_and_day_plan(
    evaluator: Evaluator,
    parent_node,
    extracted: VisitPlanExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Penguin_Event_and_Day_Plan",
        desc="Selects appropriate visit day(s) based on Wee Waddle schedule and includes complete event details",
        parent=parent_node,
        critical=True,
    )

    # Visit_Day_Selection (critical)
    days = extracted.day_selection.visit_days if extracted.day_selection else []
    visit_day_selection = evaluator.add_custom_node(
        result=_days_subset_of_thu_to_sun(days),
        id="Visit_Day_Selection",
        desc="Specifies which day(s) of the week the family should visit, and the selected day(s) are within Thu/Fri/Sat/Sun",
        parent=node,
        critical=True,
    )

    # Visit_Day_Justification (critical)
    justification_text = (
        extracted.day_selection.justification_text if extracted.day_selection else None
    ) or ""
    day_just_leaf = evaluator.add_leaf(
        id="Visit_Day_Justification",
        desc="Justifies the chosen day(s) explicitly by referencing Wee Waddle operating days",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The following text explicitly justifies the chosen visit day(s) by referencing the Wee Waddle operating days (e.g., Thu–Sun / weekends): '{justification_text}'",
        node=day_just_leaf,
        additional_instruction="Look for explicit references such as 'Wee Waddle runs Thu–Sun', 'on weekends', or similar that connect the day choice to the event schedule."
    )

    # Wee Waddle Event Details (as a critical parallel bundle with sub-checks)
    event = extracted.event_details or EventDetails()
    ww_node = evaluator.add_parallel(
        id="Wee_Waddle_Event_Details",
        desc="States Wee Waddle event name and its schedule (Thu–Sun), start time (2:15 PM), and duration (~45 minutes)",
        parent=node,
        critical=True,
    )

    # Gate: at least one source provided for event details (critical)
    ww_sources_present = evaluator.add_custom_node(
        result=bool(event.sources),
        id="Wee_Waddle_Sources_Provided",
        desc="At least one source URL provided for Wee Waddle event details",
        parent=ww_node,
        critical=True,
    )

    # Name
    name_leaf = evaluator.add_leaf(
        id="Wee_Waddle_Name_Correct",
        desc="Wee Waddle event name is correct",
        parent=ww_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The penguin event at Edinburgh Zoo is called '{GROUND_TRUTH['wee_waddle']['name']}'.",
        node=name_leaf,
        sources=event.sources,
        additional_instruction="Allow for minor punctuation/casing variants. Verify the official event name on the provided page(s)."
    )

    # Days (Thu–Sun)
    days_leaf = evaluator.add_leaf(
        id="Wee_Waddle_Days_Correct",
        desc="Wee Waddle runs Thu–Sun",
        parent=ww_node,
        critical=True,
    )
    days_phrase = ", ".join(GROUND_TRUTH["wee_waddle"]["days"])
    await evaluator.verify(
        claim=f"The Wee Waddle event runs on {days_phrase}.",
        node=days_leaf,
        sources=event.sources,
        additional_instruction="Confirm the operating days; allow variants like 'Thursday to Sunday' or 'Thu–Sun'."
    )

    # Start time 2:15 PM
    start_leaf = evaluator.add_leaf(
        id="Wee_Waddle_Start_Time_Correct",
        desc="Wee Waddle starts at 2:15 PM",
        parent=ww_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Wee Waddle starts at 2:15 PM.",
        node=start_leaf,
        sources=event.sources,
        additional_instruction="Allow time formatting variations like '2.15pm' or '14:15'."
    )

    # Duration ~45 minutes
    dur_leaf = evaluator.add_leaf(
        id="Wee_Waddle_Duration_Correct",
        desc="Wee Waddle lasts about 45 minutes",
        parent=ww_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Wee Waddle lasts about 45 minutes.",
        node=dur_leaf,
        sources=event.sources,
        additional_instruction="Allow approximations such as '~45 minutes' or 'around 45 minutes'."
    )


async def verify_unique_animals(
    evaluator: Evaluator,
    parent_node,
    extracted: VisitPlanExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Unique_Animals_Prioritized",
        desc="Prioritizes exactly the three required unique animal types and states what makes each special using constraint-grounded facts",
        parent=parent_node,
        critical=True,
    )

    # Queensland Koalas
    ko = extracted.koalas or KoalaInfo()
    ko_node = evaluator.add_parallel(
        id="Queensland_Koalas",
        desc="Includes Queensland koalas Talara and Myaree, and states they are the only Queensland koalas in the United Kingdom",
        parent=node,
        critical=True,
    )

    # Existence of names: Talara & Myaree
    ko_names_exist = evaluator.add_custom_node(
        result=set(name.strip().lower() for name in ko.names) >= {"talara", "myaree"},
        id="Koalas_Talara_Myaree_Included",
        desc="Plan includes the koalas named Talara and Myaree",
        parent=ko_node,
        critical=True,
    )

    # Mentions Queensland subspecies explicitly in the plan text
    subspecies_text = ko.subspecies_text or ""
    ko_subspecies_leaf = evaluator.add_leaf(
        id="Koalas_Queensland_Subspecies_In_Text",
        desc="Plan explicitly states they are Queensland koalas",
        parent=ko_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The following text explicitly indicates the koalas are Queensland koalas: '{subspecies_text}'",
        node=ko_subspecies_leaf,
        additional_instruction="Accept explicit references like 'Queensland koalas' or equivalent wording."
    )

    # Only-in-UK claim supported by sources
    ko_only_uk_leaf = evaluator.add_leaf(
        id="Koalas_Only_Queensland_in_UK_Supported",
        desc="Claim that Talara and Myaree are the only Queensland koalas in the UK is supported by cited sources",
        parent=ko_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Talara and Myaree are the only Queensland koalas in the United Kingdom.",
        node=ko_only_uk_leaf,
        sources=ko.sources,
        additional_instruction="Confirm this uniqueness claim on the provided official pages; allow minor wording variants."
    )

    # Red Pandas
    rp = extracted.red_pandas or RedPandaInfo()
    rp_node = evaluator.add_parallel(
        id="Red_Pandas",
        desc="Includes red pandas as a priority and states they are located behind Penguins Rock",
        parent=node,
        critical=True,
    )

    # Included in plan
    rp_included = evaluator.add_custom_node(
        result=bool((rp.mentioned_text or "").strip()),
        id="Red_Pandas_Included",
        desc="Red pandas are included as a prioritized animal in the plan",
        parent=rp_node,
        critical=True,
    )

    # Location behind Penguins Rock supported
    rp_location_leaf = evaluator.add_leaf(
        id="Red_Pandas_Location_Behind_Penguins_Rock_Supported",
        desc="States and supports that red pandas are behind Penguins Rock",
        parent=rp_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The red panda habitat at Edinburgh Zoo is behind Penguins Rock.",
        node=rp_location_leaf,
        sources=rp.sources,
        additional_instruction="Allow minor variations such as 'Penguins’ Rock' vs 'Penguins Rock'."
    )

    # Nubian Giraffes
    gf = extracted.giraffes or GiraffeInfo()
    gf_node = evaluator.add_parallel(
        id="Nubian_Giraffes",
        desc="Includes Nubian giraffes as a priority and states there are four individuals named Arrow, Gerald, Fennessy, and Gilbert",
        parent=node,
        critical=True,
    )

    # Included in plan (subspecies mentioned)
    gf_included = evaluator.add_custom_node(
        result=bool((gf.subspecies_text or "").strip()) or bool(gf.names),
        id="Giraffes_Included",
        desc="Nubian giraffes are included as a prioritized animal in the plan",
        parent=gf_node,
        critical=True,
    )

    # Names + count supported by sources
    gf_names_leaf = evaluator.add_leaf(
        id="Giraffe_Four_Names_Supported",
        desc="There are four Nubian giraffes named Arrow, Gerald, Fennessy, and Gilbert (supported)",
        parent=gf_node,
        critical=True,
    )
    await evaluator.verify(
        claim="There are four Nubian giraffes at Edinburgh Zoo named Arrow, Gerald, Fennessy, and Gilbert.",
        node=gf_names_leaf,
        sources=gf.sources,
        additional_instruction="Verify both the count (four) and the specific names on the provided page(s)."
    )


async def verify_timing_and_duration(
    evaluator: Evaluator,
    parent_node,
    extracted: VisitPlanExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Timing_and_Duration",
        desc="Provides arrival time and total visit duration, incorporating timing optimization guidance",
        parent=parent_node,
        critical=True,
    )

    tm = extracted.timing or TimingInfo()

    # Recommended Arrival Time (must be a specific time)
    arrival_ok = evaluator.add_custom_node(
        result=_has_specific_time(tm.arrival_time_text),
        id="Recommended_Arrival_Time",
        desc="Provides a recommended arrival time (a specific time-of-day is stated)",
        parent=node,
        critical=True,
    )

    # Morning Optimization Guidance (set to critical to satisfy framework constraints)
    morning_leaf = evaluator.add_leaf(
        id="Morning_Optimization_Guidance",
        desc="Recommends arriving in the morning for best animal viewing/activity (timing optimization guidance)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The plan explicitly recommends morning/early arrival for better animal activity: '{tm.morning_guidance_text or ''}'",
        node=morning_leaf,
        additional_instruction="Look for 'morning', 'early', cooler times, or similar phrasing advocating earlier arrival."
    )

    # Total Visit Duration existence
    duration_present = evaluator.add_custom_node(
        result=bool((tm.total_duration_text or "").strip()),
        id="Total_Visit_Duration",
        desc="Provides a total visit duration estimate",
        parent=node,
        critical=True,
    )

    # Duration within 3–5 hours (set to critical for consistency under critical parent)
    within_3_5_leaf = evaluator.add_leaf(
        id="Duration_3_to_5_Hours",
        desc="Keeps the recommended total duration within 3–5 hours",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The stated total visit duration '{tm.total_duration_text or ''}' is within 3 to 5 hours inclusive.",
        node=within_3_5_leaf,
        additional_instruction="Interpret ranges like '3–5 hours' as within bounds; if multiple durations are mentioned, judge the main/explicit recommendation."
    )

    # Zoo size fact: 82 acres (critical; verify with sources)
    size_leaf = evaluator.add_leaf(
        id="Zoo_Size_Fact",
        desc="States that Edinburgh Zoo covers 82 acres (or explicitly uses this fact in timing/duration justification)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Edinburgh Zoo covers about 82 acres.",
        node=size_leaf,
        sources=tm.zoo_size_sources,
        additional_instruction="Allow approximate phrasing and metric conversions (about 33 hectares). Verify with the provided official sources."
    )


async def verify_lunch_break_facilities(
    evaluator: Evaluator,
    parent_node,
    extracted: VisitPlanExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Lunch_Break_Facilities",
        desc="Plans a lunch break using the zoo’s on-site facilities",
        parent=parent_node,
        critical=True,
    )

    ln = extracted.lunch or LunchInfo()

    # Explicit lunch break included
    lunch_in_plan = evaluator.add_custom_node(
        result=bool(ln.lunch_in_plan) or bool(ln.facilities_mentioned),
        id="Lunch_Break_In_Plan",
        desc="Includes an explicit lunch break in the itinerary",
        parent=node,
        critical=True,
    )

    # Picnic areas identified (create a parallel subgroup to check text + source support)
    picnic_node = evaluator.add_parallel(
        id="Picnic_Areas_Identified",
        desc="Identifies appropriate picnic facilities/areas for lunch (hilltop and/or main lawn areas)",
        parent=node,
        critical=True,
    )

    # Text includes hilltop and/or main lawn
    facilities = ln.facilities_mentioned or []
    text_has_required = evaluator.add_custom_node(
        result=_list_contains_any(facilities, ["hilltop", "main lawn"]),
        id="Picnic_Areas_In_Text",
        desc="Plan text identifies 'hilltop' and/or 'main lawn' picnic areas",
        parent=picnic_node,
        critical=True,
    )

    # Source support that these are picnic facilities/areas
    picnic_src_leaf = evaluator.add_leaf(
        id="Picnic_Areas_Supported",
        desc="Hilltop and/or main lawn picnic areas are supported by cited sources",
        parent=picnic_node,
        critical=True,
    )
    # Determine which claim to verify based on extracted facilities
    claim_phrase = "Edinburgh Zoo provides picnic areas at the hilltop and/or main lawn."
    await evaluator.verify(
        claim=claim_phrase,
        node=picnic_src_leaf,
        sources=ln.sources,
        additional_instruction="Any official page/map that mentions picnic areas at the 'hilltop' and/or 'main lawn' suffices."
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
    # Initialize evaluator with a neutral root, then attach our critical Visit_Plan node under it
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

    # Extract structured plan
    extracted_plan = await evaluator.extract(
        prompt=prompt_extract_visit_plan(),
        template_class=VisitPlanExtraction,
        extraction_name="visit_plan_extraction"
    )

    # Add ground truth info useful for debugging
    evaluator.add_ground_truth({
        "wee_waddle_expected": GROUND_TRUTH["wee_waddle"],
        "unique_animals_expected": GROUND_TRUTH["unique_animals"],
        "zoo_size_expected_acres": GROUND_TRUTH["zoo_size_acres"]
    }, gt_type="expected_facts")

    # Build the Visit_Plan critical node
    visit_plan_node = evaluator.add_parallel(
        id="Visit_Plan",
        desc="Complete Edinburgh Zoo visit plan satisfying the proposed question and the listed constraints",
        parent=root,
        critical=True
    )

    # Subtrees
    await verify_penguin_event_and_day_plan(evaluator, visit_plan_node, extracted_plan)
    await verify_unique_animals(evaluator, visit_plan_node, extracted_plan)
    await verify_timing_and_duration(evaluator, visit_plan_node, extracted_plan)
    await verify_lunch_break_facilities(evaluator, visit_plan_node, extracted_plan)

    # Return evaluation summary
    return evaluator.get_summary()