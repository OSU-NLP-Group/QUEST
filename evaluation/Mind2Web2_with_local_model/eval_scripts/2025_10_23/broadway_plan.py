import asyncio
import logging
from datetime import datetime, date
from typing import Optional, List, Dict

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.api_tools import tool_googlemap

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broadway_plan"
TASK_DESCRIPTION = """
I am going to spend three days in New York City on 10th-12th of next month. I want to attend one Broadway show each day. Please check the available events during those days and make a plan for me. For each event, please list its name, specific scheduling time and theater location. Besides, for each show, please also find a subway station within 10min walking distance to the theater and provide the station name.
"""

# Calculate the actual dates (10th-12th of next month)
def get_next_month_dates():
    today = datetime.utcnow().date()

    if today.month == 12:
        next_month = 1
        year = today.year + 1
    else:
        next_month = today.month + 1
        year = today.year

    return [
        date(year, next_month, day).strftime("%Y-%m-%d")
        for day in range(10, 13)
    ]

TARGET_DATES = get_next_month_dates()

# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class ShowBasicInfo(BaseModel):
    """Basic show information for initial extraction."""
    show_name: Optional[str] = Field(default=None)
    date: Optional[str] = Field(default=None)


class ShowsOverview(BaseModel):
    """Overview of all three days' shows."""
    day1: ShowBasicInfo = Field(default_factory=ShowBasicInfo)
    day2: ShowBasicInfo = Field(default_factory=ShowBasicInfo)
    day3: ShowBasicInfo = Field(default_factory=ShowBasicInfo)


class ShowDetailedInfo(BaseModel):
    """Detailed information for a specific show."""
    time: Optional[str] = Field(default=None)
    theater_location: Optional[str] = Field(default=None)
    subway_station: Optional[str] = Field(default=None)
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_shows_overview() -> str:
    return f"""
    Extract the basic information about Broadway shows planned for the three days.
    
    For each day (Day 1: {TARGET_DATES[0]}, Day 2: {TARGET_DATES[1]}, Day 3: {TARGET_DATES[2]} - dates are in YYYY-MM-DD format), extract:
    - show_name: The name of the Broadway show
    - date: The date as mentioned in the answer
    
    Return null if information is not provided for a particular day.
    """


def prompt_extract_show_details(show_name: str, date: str, day_num: int) -> str:
    return f"""
    For the Broadway show "{show_name}" scheduled on {date} (Day {day_num}), extract the following detailed information:
    
    - time: The specific showtime (e.g., "7:30 PM", "2:00 PM")
    - theater_location: The theater location/address
    - subway_station: The specific subway station name within 10min walking distance
    - urls: ALL URLs provided that relate to this show or theater
    
    Return null for any field not clearly provided in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification functions for individual shows                                 #
# --------------------------------------------------------------------------- #
async def verify_show_requirements(
        evaluator: Evaluator,
        show_basic: ShowBasicInfo,
        show_details: ShowDetailedInfo,
        day_num: int,
        expected_date: str,
        gmaps_tool: tool_googlemap.GoogleMapsTool,
) -> None:
    """
    Verify all requirements for a single day's Broadway show.
    """
    show_node = evaluator.add_parallel(
        id=f"day_{day_num}_show",
        desc=f"Day {day_num} Broadway show meets all requirements",
        critical=False  # Allow partial scoring across days
    )

    # 1. Verify basic information is provided
    await verify_basic_show_info(evaluator, show_node, show_basic, show_details, day_num)
    
    # 2. Verify show details from URLs
    await verify_show_details(evaluator, show_node, show_basic, show_details, day_num)
    
    # 3. Verify date matches expected
    await verify_show_date(evaluator, show_node, show_basic, day_num, expected_date)
    
    # 4. Verify theater location
    await verify_theater_location(evaluator, show_node, show_basic, show_details, day_num)
    
    # 5. Verify subway station requirements
    await verify_subway_station(evaluator, show_node, show_details, day_num, gmaps_tool)


async def verify_basic_show_info(
        evaluator: Evaluator,
        parent_node,
        show_basic: ShowBasicInfo,
        show_details: ShowDetailedInfo,
        day_num: int,
) -> None:
    """Verify that all required basic information is provided."""
    
    has_all_info = bool(
        show_basic.show_name and
        show_basic.date and
        show_details.time and
        show_details.theater_location and
        show_details.subway_station and
        show_details.urls
    )
    
    basic_info_node = evaluator.add_custom_node(
        result=has_all_info,
        id=f"day_{day_num}_basic_info",
        desc="All required information (show name, date, time, theater location, subway station, URLs) is provided",
        parent=parent_node,
        critical=True,
    )


async def verify_show_details(
        evaluator: Evaluator,
        parent_node,
        show_basic: ShowBasicInfo,
        show_details: ShowDetailedInfo,
        day_num: int,
) -> None:
    """Verify that the show details are substantiated by the provided URLs."""
    
    show_details_node = evaluator.add_leaf(
        id=f"day_{day_num}_show_details",
        desc="Show name and schedule are substantiated by provided sources",
        parent=parent_node,
        critical=True,
    )
    
    claim = f"The Broadway show '{show_basic.show_name}' is scheduled at {show_details.time} on {show_basic.date}"
    await evaluator.verify(
        claim=claim,
        node=show_details_node,
        sources=show_details.urls,
        additional_instruction="Verify that the show name, date, and time match the information on the provided pages. The information might appear under sections such as 'schedules', 'upcoming schedules', 'calendar', and 'show times'. Prioritize carefully examining the web text extracted from the webpage than the screenshot."
    )


async def verify_show_date(
        evaluator: Evaluator,
        parent_node,
        show_basic: ShowBasicInfo,
        day_num: int,
        expected_date: str,
) -> None:
    """Verify that the show date matches the expected date."""
    
    date_match_node = evaluator.add_leaf(
        id=f"day_{day_num}_date_match",
        desc=f"Show is scheduled for the correct date (Day {day_num})",
        parent=parent_node,
        critical=True,
    )
    claim = f"The show date '{show_basic.date}' matches or is equivalent to '{expected_date}' for Day {day_num} of the trip"
    await evaluator.verify(
        claim=claim,
        node=date_match_node,
        additional_instruction=f"Check if the dates match, allowing for different date formats. The expected date '{expected_date}' is in YYYY-MM-DD format."
    )


async def verify_theater_location(
        evaluator: Evaluator,
        parent_node,
        show_basic: ShowBasicInfo,
        show_details: ShowDetailedInfo,
        day_num: int,
) -> None:
    """Verify that the theater location is accurate."""
    
    theater_location_node = evaluator.add_leaf(
        id=f"day_{day_num}_theater_location",
        desc="Theater location is verified",
        parent=parent_node,
        critical=True,
    )
    
    claim = f"The page shows that the theater hosting '{show_basic.show_name}' is located at '{show_details.theater_location}'"
    await evaluator.verify(
        claim=claim,
        node=theater_location_node,
        sources=show_details.urls,
        additional_instruction="Verify that the theater location/address on the provided pages matches the stated location. Look for the theater address on official show pages, theater websites, or ticketing platforms."
    )


async def verify_subway_station(
        evaluator: Evaluator,
        parent_node,
        show_details: ShowDetailedInfo,
        day_num: int,
        gmaps_tool: tool_googlemap.GoogleMapsTool,
) -> None:
    """Verify that the subway station is within 10 minutes walking distance."""
    
    # First create a parallel node to hold both checks
    subway_verification_node = evaluator.add_parallel(
        id=f"day_{day_num}_subway_verification",
        desc="Subway station verification",
        parent=parent_node,
        critical=False,
    )
    
    # Check 1: Verify it's a real NYC subway station
    subway_validity_node = evaluator.add_leaf(
        id=f"day_{day_num}_subway_validity",
        desc="Subway station is a valid NYC subway station",
        parent=subway_verification_node,
        critical=True,
    )
    
    claim = f"'{show_details.subway_station}' is a valid New York City subway station name"
    await evaluator.verify(
        claim=claim,
        node=subway_validity_node,
        additional_instruction="Check if this follows NYC subway station naming patterns. Common formats include: street names/numbers (e.g., '42nd St', 'Times Sq-42 St'), location names (e.g., 'Columbus Circle'), or avenue/street combinations. Accept any reasonable subway station name format for NYC."
    )

    # Check 2: Verify walking distance using add_custom_node
    if subway_validity_node.status == "passed":
        try:
            # Calculate walking time from subway to theater
            walking_time_seconds = await gmaps_tool.calculate_travel_time(
                f"{show_details.subway_station} subway station, New York, NY",
                show_details.theater_location,
                mode="walking"
            )
            
            if isinstance(walking_time_seconds, int):
                walking_time_minutes = walking_time_seconds / 60
                is_within_10_min = walking_time_minutes <= 10.0
                
                distance_node = evaluator.add_custom_node(
                    result=is_within_10_min,
                    id=f"day_{day_num}_subway_distance",
                    desc=f"Subway station is within 10min walking distance ({walking_time_minutes:.1f} min)",
                    parent=subway_verification_node,
                    critical=True,
                )
            else:
                # API call returned non-integer
                evaluator.add_custom_node(
                    result=False,
                    id=f"day_{day_num}_subway_distance",
                    desc=f"Google Maps API returned invalid response: {walking_time_seconds}",
                    parent=subway_verification_node,
                    critical=True,
                )
        except Exception as e:
            # Exception occurred
            evaluator.add_custom_node(
                result=False,
                id=f"day_{day_num}_subway_distance",
                desc=f"Error calculating walking distance: {type(e).__name__}: {str(e)}",
                parent=subway_verification_node,
                critical=True,
            )
    else:
        evaluator.add_custom_node(
                result=False,
                id=f"day_{day_num}_subway_distance",
                desc=f"skipped due to the failure in subway_validity_node",
                parent=subway_verification_node,
                critical=True,
            )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Initialize evaluator -------- #
    evaluator = Evaluator()
    evaluator.initialize(
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

    # -------- 2. Set up Google Maps tool -------- #
    gmaps_tool = tool_googlemap.GoogleMapsTool()

    # -------- 3. Extract shows overview first -------- #
    shows_overview = await evaluator.extract(
        prompt=prompt_extract_shows_overview(),
        template_class=ShowsOverview,
        extraction_name="shows_overview"
    )

    # -------- 4. Extract detailed info for each show and verify -------- #
    shows_basic = [
        (shows_overview.day1, 1, TARGET_DATES[0]),
        (shows_overview.day2, 2, TARGET_DATES[1]),
        (shows_overview.day3, 3, TARGET_DATES[2]),
    ]

    for show_basic, day_num, expected_date in shows_basic:
        # Extract detailed info only if we have a show name
        if show_basic.show_name:
            # Use the date from extraction if available, otherwise use expected date
            date_for_prompt = show_basic.date if show_basic.date else expected_date

            show_details = await evaluator.extract(
                prompt=prompt_extract_show_details(
                    show_basic.show_name,
                    date_for_prompt,
                    day_num
                ),
                template_class=ShowDetailedInfo,
                extraction_name=f"day_{day_num}_show_details"
            )
        else:
            # Create empty details for missing show
            show_details = ShowDetailedInfo()

        # Verify this show
        await verify_show_requirements(
            evaluator=evaluator,
            show_basic=show_basic,
            show_details=show_details,
            day_num=day_num,
            expected_date=expected_date,
            gmaps_tool=gmaps_tool,
        )

    # -------- 5. Add custom info -------- #
    evaluator.add_custom_info({
        "target_dates": TARGET_DATES,
        "shows_found": sum(1 for s in [shows_overview.day1, shows_overview.day2, shows_overview.day3] if s.show_name),
    }, "task_statistics")
    
    # -------- 6. Return structured result -------- #
    return evaluator.get_summary()
