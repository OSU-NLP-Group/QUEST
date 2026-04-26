import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chicago_marathon_2026_comprehensive_info"
TASK_DESCRIPTION = (
    "Provide comprehensive race-day information for the 2026 Bank of America Chicago Marathon. "
    "Your answer must include all of the following 14 specific details: "
    "(1) the official race date (month, day, and year), "
    "(2) the day of the week the race takes place, "
    "(3) the complete official name of the race, "
    "(4) the start location, "
    "(5) the finish location, "
    "(6) the host city, "
    "(7) the start time for professional runners, "
    "(8) the start time for Wave 1 participants, "
    "(9) the course time limit, "
    "(10) the approximate maximum field size or capacity, "
    "(11) the minimum age requirement for participants, "
    "(12) the World Athletics certification level held by the race, "
    "(13) the total number of aid stations positioned along the course, and "
    "(14) the official race website URL."
)


# Ground truth expectations from the rubric (for reference in summary only)
GROUND_TRUTH = {
    "official_race_date": "October 11, 2026",
    "day_of_week": "Sunday",
    "official_race_name": "Bank of America Chicago Marathon",
    "start_location": "Grant Park",
    "finish_location": "Grant Park",
    "host_city": "Chicago",
    "professional_start_time": "7:30 AM",
    "wave1_start_time": "7:35 AM",
    "course_time_limit": "6 hours and 30 minutes",
    "field_size": "55,000 runners",
    "minimum_age": "16 years old",
    "world_athletics_cert": "World Athletics Platinum Label",
    "aid_stations_count": "20",
    "official_website_url": "https://www.chicagomarathon.com",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FieldValue(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MarathonInfo(BaseModel):
    official_race_date: Optional[FieldValue] = None
    day_of_week: Optional[FieldValue] = None
    official_race_name: Optional[FieldValue] = None
    start_location: Optional[FieldValue] = None
    finish_location: Optional[FieldValue] = None
    host_city: Optional[FieldValue] = None
    professional_start_time: Optional[FieldValue] = None
    wave1_start_time: Optional[FieldValue] = None
    course_time_limit: Optional[FieldValue] = None
    field_size: Optional[FieldValue] = None
    minimum_age: Optional[FieldValue] = None
    world_athletics_cert: Optional[FieldValue] = None
    aid_stations_count: Optional[FieldValue] = None
    official_website_url: Optional[FieldValue] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_marathon_info() -> str:
    return """
    Extract the following 14 items exactly as they appear in the provided answer, along with all URLs the answer cites as evidence for each item. For each item, return a JSON object with fields: 
    - value: the exact value text as stated in the answer
    - sources: an array of all URLs that the answer explicitly cites for this item (include full URLs; if missing protocol, prepend http://)

    Items to extract (use these exact JSON keys):
    1) official_race_date: The official 2026 race date (e.g., "October 11, 2026")
    2) day_of_week: The day of the week (e.g., "Sunday")
    3) official_race_name: The complete official name of the race (e.g., "Bank of America Chicago Marathon")
    4) start_location: The official start location
    5) finish_location: The official finish location
    6) host_city: The host city name
    7) professional_start_time: The start time for professional/elite runners (local Chicago time)
    8) wave1_start_time: The start time for Wave 1 participants (local Chicago time)
    9) course_time_limit: The official course time limit (e.g., "6 hours and 30 minutes")
    10) field_size: The approximate maximum field size or capacity (e.g., "55,000 runners" or "55,000")
    11) minimum_age: The minimum age requirement on race day (e.g., "16 years old")
    12) world_athletics_cert: The World Athletics certification level (e.g., "World Athletics Platinum Label")
    13) aid_stations_count: The total number of aid stations along the course (e.g., "20")
    14) official_website_url: The official race website URL. For this field, if the answer gives a URL, also include that URL in the 'sources' array even if no extra source is cited.

    Rules:
    - Do not invent values or URLs. If the answer omits an item, set its 'value' to null and 'sources' to [].
    - For URLs embedded in markdown links, extract the actual link target URL.
    - Keep values as strings; do not convert formats (e.g., keep "6 hours and 30 minutes" as-is).
    - If multiple URLs are cited for one item, include all of them in 'sources'.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _norm_sources(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def add_fact_check_group(
    evaluator: Evaluator,
    parent_node,
    *,
    group_id: str,
    verify_leaf_id: str,
    group_desc: str,
    verify_leaf_desc: str,
    field: Optional[FieldValue],
    claim_text: str,
    sources_list: List[str],
    additional_instruction: str = "None",
    extra_prerequisites: Optional[List[Any]] = None
):
    """
    Build a critical sequential group for one fact:
    1) value_present (custom, critical)
    2) sources_present (custom, critical)
    3) supported_by_sources (verify leaf, critical)
    """
    # Critical group node to enforce all checks
    group_node = evaluator.add_sequential(
        id=f"{group_id}_main",
        desc=group_desc,
        parent=parent_node,
        critical=True
    )

    value_present = bool(field and field.value and str(field.value).strip())
    sources_present_flag = bool(_norm_sources(sources_list))

    evaluator.add_custom_node(
        result=value_present,
        id=f"{group_id}_value_present",
        desc=f"{group_desc} - value is provided in the answer",
        parent=group_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=sources_present_flag,
        id=f'{group_id}_sources_present',
        desc=f"{group_desc} - at least one source URL is provided",
        parent=group_node,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id=verify_leaf_id,
        desc=verify_leaf_desc,
        parent=group_node,
        critical=True
    )

    # If preconditions fail, verify() will be auto-skipped by the evaluator
    await evaluator.verify(
        claim=claim_text,
        node=verify_leaf,
        sources=_norm_sources(sources_list),
        additional_instruction=additional_instruction,
        extra_prerequisites=extra_prerequisites
    )

    return {
        "group_node": group_node,
        "verify_leaf": verify_leaf
    }


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the comprehensive 2026 Bank of America Chicago Marathon info task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # children independent at top level
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

    # Add a critical aggregator node under root to enforce "all must pass"
    agg_node = evaluator.add_parallel(
        id="Comprehensive_Chicago_Marathon_2026_Information",
        desc="Complete race-day information for the 2026 Bank of America Chicago Marathon",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extracted: MarathonInfo = await evaluator.extract(
        prompt=prompt_extract_marathon_info(),
        template_class=MarathonInfo,
        extraction_name="marathon_info_extraction"
    )

    # Add ground-truth (reference) info for transparency (not used to grade directly)
    evaluator.add_ground_truth(
        {
            "expected": GROUND_TRUTH,
            "note": "These are expectations from the rubric; grading is primarily source-grounded."
        },
        gt_type="rubric_reference"
    )

    # Build groups
    # 1) Official Race Date
    race_date_sources = _norm_sources(extracted.official_race_date.sources if extracted.official_race_date else [])
    race_date_val = extracted.official_race_date.value if extracted.official_race_date else None
    res_date = await add_fact_check_group(
        evaluator,
        agg_node,
        group_id="Official_Race_Date",
        verify_leaf_id="Official_Race_Date",
        group_desc="Official race date",
        verify_leaf_desc="The race takes place on October 11, 2026",
        field=extracted.official_race_date,
        claim_text=f"The official race date for the 2026 Bank of America Chicago Marathon is {race_date_val}.",
        sources_list=race_date_sources,
        additional_instruction="Accept minor formatting variations (e.g., inclusion of weekday). The statement must clearly match the 2026 event date as presented on the cited page(s)."
    )
    date_supported_leaf = res_date["verify_leaf"]

    # 2) Day of Week (allow inference from date if needed)
    dow_sources = _norm_sources(
        (extracted.day_of_week.sources if (extracted.day_of_week and extracted.day_of_week.sources) else race_date_sources)
    )
    dow_val = extracted.day_of_week.value if extracted.day_of_week else None
    await add_fact_check_group(
        evaluator,
        agg_node,
        group_id="Day_of_Week",
        verify_leaf_id="Day_of_Week",
        group_desc="Race weekday",
        verify_leaf_desc="The race takes place on a Sunday",
        field=extracted.day_of_week,
        claim_text=f"The 2026 Bank of America Chicago Marathon takes place on a {dow_val}.",
        sources_list=dow_sources,
        additional_instruction="If the webpage gives the official date (e.g., October 11, 2026) but not the weekday text, it is acceptable to infer the weekday via calendar logic.",
        extra_prerequisites=[date_supported_leaf]  # depend on date support
    )

    # 3) Official Race Name
    orn_sources = _norm_sources(extracted.official_race_name.sources if extracted.official_race_name else [])
    orn_val = extracted.official_race_name.value if extracted.official_race_name else None
    await add_fact_check_group(
        evaluator,
        agg_node,
        group_id="Official_Race_Name",
        verify_leaf_id="Official_Race_Name",
        group_desc="Official race name",
        verify_leaf_desc="The official name is 'Bank of America Chicago Marathon'",
        field=extracted.official_race_name,
        claim_text=f"The complete official name of the event is {orn_val}.",
        sources_list=orn_sources,
        additional_instruction="Require the full sponsored name (Bank of America Chicago Marathon). Do not accept shortened forms like 'Chicago Marathon' alone."
    )

    # 4) Start Location
    start_sources = _norm_sources(extracted.start_location.sources if extracted.start_location else [])
    start_val = extracted.start_location.value if extracted.start_location else None
    await add_fact_check_group(
        evaluator,
        agg_node,
        group_id="Start_Location",
        verify_leaf_id="Start_Location",
        group_desc="Start location",
        verify_leaf_desc="The race starts in Grant Park",
        field=extracted.start_location,
        claim_text=f"The marathon start location is {start_val}.",
        sources_list=start_sources,
        additional_instruction="Accept 'Grant Park' or equivalent official phrasing (e.g., specific streets within Grant Park)."
    )

    # 5) Finish Location
    finish_sources = _norm_sources(extracted.finish_location.sources if extracted.finish_location else [])
    finish_val = extracted.finish_location.value if extracted.finish_location else None
    await add_fact_check_group(
        evaluator,
        agg_node,
        group_id="Finish_Location",
        verify_leaf_id="Finish_Location",
        group_desc="Finish location",
        verify_leaf_desc="The race finishes in Grant Park",
        field=extracted.finish_location,
        claim_text=f"The marathon finish location is {finish_val}.",
        sources_list=finish_sources,
        additional_instruction="Accept 'Grant Park' or equivalent official phrasing (e.g., specific finish area within Grant Park)."
    )

    # 6) Host City
    city_sources = _norm_sources(extracted.host_city.sources if extracted.host_city else [])
    city_val = extracted.host_city.value if extracted.host_city else None
    await add_fact_check_group(
        evaluator,
        agg_node,
        group_id="Host_City",
        verify_leaf_id="Host_City",
        group_desc="Host city",
        verify_leaf_desc="The race takes place in Chicago",
        field=extracted.host_city,
        claim_text=f"The host city is {city_val}.",
        sources_list=city_sources,
        additional_instruction="Allow 'Chicago' or 'Chicago, Illinois' as equivalent."
    )

    # 7) Professional Start Time
    pst_sources = _norm_sources(extracted.professional_start_time.sources if extracted.professional_start_time else [])
    pst_val = extracted.professional_start_time.value if extracted.professional_start_time else None
    await add_fact_check_group(
        evaluator,
        agg_node,
        group_id="Professional_Start_Time",
        verify_leaf_id="Professional_Start_Time",
        group_desc="Professional/elite start time",
        verify_leaf_desc="The professional start time is 7:30 AM",
        field=extracted.professional_start_time,
        claim_text=f"The professional (elite) start time is {pst_val} local time in Chicago (Central Time).",
        sources_list=pst_sources,
        additional_instruction="Accept reasonable formatting variants (e.g., 7:30 a.m., 07:30). The time should apply to the elite/professional start."
    )

    # 8) Wave 1 Start Time
    w1_sources = _norm_sources(extracted.wave1_start_time.sources if extracted.wave1_start_time else [])
    w1_val = extracted.wave1_start_time.value if extracted.wave1_start_time else None
    await add_fact_check_group(
        evaluator,
        agg_node,
        group_id="Wave_1_Start_Time",
        verify_leaf_id="Wave_1_Start_Time",
        group_desc="Wave 1 start time",
        verify_leaf_desc="Wave 1 starts at 7:35 AM",
        field=extracted.wave1_start_time,
        claim_text=f"The Wave 1 start time is {w1_val} local time in Chicago (Central Time).",
        sources_list=w1_sources,
        additional_instruction="Accept reasonable formatting variants (e.g., 7:35 a.m., 07:35). The time should be specific to Wave 1."
    )

    # 9) Course Time Limit
    ctl_sources = _norm_sources(extracted.course_time_limit.sources if extracted.course_time_limit else [])
    ctl_val = extracted.course_time_limit.value if extracted.course_time_limit else None
    await add_fact_check_group(
        evaluator,
        agg_node,
        group_id="Course_Time_Limit",
        verify_leaf_id="Course_Time_Limit",
        group_desc="Course time limit",
        verify_leaf_desc="The course time limit is 6 hours and 30 minutes",
        field=extracted.course_time_limit,
        claim_text=f"The official course time limit to complete the marathon is {ctl_val}.",
        sources_list=ctl_sources,
        additional_instruction="Accept standard equivalents like '6:30:00' for '6 hours and 30 minutes'."
    )

    # 10) Field Size
    fs_sources = _norm_sources(extracted.field_size.sources if extracted.field_size else [])
    fs_val = extracted.field_size.value if extracted.field_size else None
    await add_fact_check_group(
        evaluator,
        agg_node,
        group_id="Field_Size",
        verify_leaf_id="Field_Size",
        group_desc="Field size / capacity",
        verify_leaf_desc="The approximate field size is 55,000 runners",
        field=extracted.field_size,
        claim_text=f"The approximate maximum field size/capacity is {fs_val}.",
        sources_list=fs_sources,
        additional_instruction="Approximate or 'up to' phrasing is acceptable as long as it semantically matches the supplied figure."
    )

    # 11) Minimum Age Requirement
    ma_sources = _norm_sources(extracted.minimum_age.sources if extracted.minimum_age else [])
    ma_val = extracted.minimum_age.value if extracted.minimum_age else None
    await add_fact_check_group(
        evaluator,
        agg_node,
        group_id="Minimum_Age_Requirement",
        verify_leaf_id="Minimum_Age_Requirement",
        group_desc="Minimum age requirement",
        verify_leaf_desc="Participants must be at least 16 years old on race day",
        field=extracted.minimum_age,
        claim_text=f"The minimum age requirement is {ma_val} on race day.",
        sources_list=ma_sources,
        additional_instruction="Focus on the age requirement specifically for marathon participation on race day."
    )

    # 12) World Athletics Certification
    wa_sources = _norm_sources(extracted.world_athletics_cert.sources if extracted.world_athletics_cert else [])
    wa_val = extracted.world_athletics_cert.value if extracted.world_athletics_cert else None
    await add_fact_check_group(
        evaluator,
        agg_node,
        group_id="World_Athletics_Certification",
        verify_leaf_id="World_Athletics_Certification",
        group_desc="World Athletics certification level",
        verify_leaf_desc="The race holds World Athletics Platinum Label certification",
        field=extracted.world_athletics_cert,
        claim_text=f"The event holds the World Athletics certification level: {wa_val}.",
        sources_list=wa_sources,
        additional_instruction="Confirm the current World Athletics label (e.g., Platinum Label) as stated on authoritative sources."
    )

    # 13) Number of Aid Stations
    as_sources = _norm_sources(extracted.aid_stations_count.sources if extracted.aid_stations_count else [])
    as_val = extracted.aid_stations_count.value if extracted.aid_stations_count else None
    await add_fact_check_group(
        evaluator,
        agg_node,
        group_id="Number_of_Aid_Stations",
        verify_leaf_id="Number_of_Aid_Stations",
        group_desc="Total number of aid stations",
        verify_leaf_desc="There are 20 aid stations along the course",
        field=extracted.aid_stations_count,
        claim_text=f"There are {as_val} aid stations positioned along the marathon course.",
        sources_list=as_sources,
        additional_instruction="Accept if the total count matches, even if the page also lists locations or types."
    )

    # 14) Official Website URL
    # For this item, use the URL value itself as the verification source if available.
    ow_field = extracted.official_website_url
    ow_val = ow_field.value if ow_field else None
    ow_sources = _norm_sources(
        ([ow_val] if ow_val else []) + (ow_field.sources if (ow_field and ow_field.sources) else [])
    )
    await add_fact_check_group(
        evaluator,
        agg_node,
        group_id="Official_Website_URL",
        verify_leaf_id="Official_Website_URL",
        group_desc="Official race website URL",
        verify_leaf_desc="The official website is www.chicagomarathon.com",
        field=ow_field,
        claim_text="This webpage is the official website of the Bank of America Chicago Marathon.",
        sources_list=ow_sources,
        additional_instruction="Verify that the cited URL is the official site for the event (look for branding, ownership, and official language on the page)."
    )

    # Return structured result
    return evaluator.get_summary()