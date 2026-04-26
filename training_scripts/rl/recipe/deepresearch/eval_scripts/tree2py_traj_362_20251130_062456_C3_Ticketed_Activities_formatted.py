import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dwts_radiocity_2026_0215"
TASK_DESCRIPTION = (
    "Provide complete details for the Dancing With The Stars: Live! - 2026 Tour show scheduled at Radio City Music Hall "
    "in New York, NY on February 15, 2026. Your answer must include: (1) confirmation of the venue and date with a supporting URL, "
    "(2) both performance times scheduled for that day, (3) the name of the Special Guest Co-Host scheduled to appear on this date "
    "with a supporting URL, (4) a list of all three VIP package tiers available for purchase along with their key features and a supporting URL, "
    "and (5) the ticket purchasing platform and direct URL to buy tickets for this specific show."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TourInfo(BaseModel):
    name: Optional[str] = None
    date_range_text: Optional[str] = None  # e.g., "January 22, 2026 - May 2, 2026"
    urls: List[str] = Field(default_factory=list)  # URLs confirming tour/date range


class VenueDateInfo(BaseModel):
    venue_name: Optional[str] = None  # e.g., "Radio City Music Hall"
    city_state: Optional[str] = None  # e.g., "New York, NY"
    show_date_text: Optional[str] = None  # e.g., "Sunday, February 15, 2026"
    urls: List[str] = Field(default_factory=list)  # URLs confirming venue and date


class PerformanceTimesInfo(BaseModel):
    matinee_time_text: Optional[str] = None  # e.g., "2:00 PM ET"
    evening_time_text: Optional[str] = None  # e.g., "8:00 PM ET"
    urls: List[str] = Field(default_factory=list)  # URLs confirming performance times


class CoHostInfo(BaseModel):
    name: Optional[str] = None  # e.g., "Danielle Fishel"
    urls: List[str] = Field(default_factory=list)  # URLs confirming co-host schedule


class VIPInfo(BaseModel):
    package_names: List[str] = Field(default_factory=list)  # Names of the three tiers
    key_features: List[str] = Field(default_factory=list)  # Features mentioned in the answer
    urls: List[str] = Field(default_factory=list)  # URLs confirming VIP details


class TicketingInfo(BaseModel):
    platform_name: Optional[str] = None  # e.g., "Ticketmaster"
    purchase_url: Optional[str] = None  # Direct URL to buy tickets for this specific show


class DWTSShowExtraction(BaseModel):
    """Complete extraction of all required fields from the answer."""
    tour: Optional[TourInfo] = None
    venue_date: Optional[VenueDateInfo] = None
    performance_times: Optional[PerformanceTimesInfo] = None
    cohost: Optional[CoHostInfo] = None
    vip: Optional[VIPInfo] = None
    ticketing: Optional[TicketingInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_dwts_show() -> str:
    return """
    Extract the required details for the Dancing With The Stars: Live! - 2026 Tour show at Radio City Music Hall, New York, NY, on February 15, 2026 from the provided answer.
    Return a JSON object matching this schema:

    {
      "tour": {
        "name": string | null,
        "date_range_text": string | null,
        "urls": string[]  // Official URLs confirming the tour or its date range (DWTS tour site, Ticketmaster, or venue)
      },
      "venue_date": {
        "venue_name": string | null,
        "city_state": string | null,
        "show_date_text": string | null,
        "urls": string[]  // Official URLs confirming the venue and date (Ticketmaster, DWTS tour site, or venue website)
      },
      "performance_times": {
        "matinee_time_text": string | null,
        "evening_time_text": string | null,
        "urls": string[]  // Official URLs confirming both performance times for Feb 15, 2026
      },
      "cohost": {
        "name": string | null,
        "urls": string[]  // Official URLs confirming the special guest co-host schedule
      },
      "vip": {
        "package_names": string[],  // List all VIP tier names mentioned
        "key_features": string[],   // Features mentioned in the answer (e.g., interactive experience, professional photo opportunity)
        "urls": string[]            // Official URLs confirming VIP package details
      },
      "ticketing": {
        "platform_name": string | null,  // e.g., Ticketmaster
        "purchase_url": string | null    // Direct URL to purchase tickets for the Feb 15, 2026 Radio City show
      }
    }

    Rules:
    - Extract ONLY what is explicitly stated in the answer.
    - For any missing field, use null (for strings) or an empty array (for lists).
    - For URLs, extract complete valid URLs that are present in the answer text (plain or markdown).
    - Do not invent or infer details.
    - Keep times and dates as written (e.g., "2:00 PM ET", "Sunday, February 15, 2026").
    - If multiple official URLs are provided, include them all in the corresponding array.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_dwts_show(
    evaluator: Evaluator,
    parent_node,
    data: DWTSShowExtraction,
) -> None:
    """
    Build the verification tree and perform checks according to the rubric.
    All nodes under the main rubric node are critical to satisfy the 'complete details' requirement.
    """

    # Top-level rubric node (critical, parallel aggregation)
    main = evaluator.add_parallel(
        id="DWTS_RadioCity_February15_Complete_Details",
        desc=(
            "Provide complete details for the DWTS: Live! - 2026 Tour show at Radio City Music Hall on February 15, 2026, "
            "including tour qualification, venue/date, performance times, special guest, VIP packages, and ticketing information, "
            "with official supporting URLs for major claims."
        ),
        parent=parent_node,
        critical=True,
    )

    # --------------------------- Tour Qualification --------------------------- #
    tour_group = evaluator.add_parallel(
        id="Tour_Qualification",
        desc="Confirm the event is part of the Dancing With The Stars: Live! - 2026 Tour and that the tour runs from Jan 22, 2026 to May 2, 2026, with an official supporting URL.",
        parent=main,
        critical=True,
    )

    # Tour Name Stated
    tour_name_leaf = evaluator.add_leaf(
        id="Tour_Name_Stated",
        desc="State that the show is part of the Dancing With The Stars: Live! - 2026 Tour.",
        parent=tour_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the show is part of the 'Dancing With The Stars: Live! - 2026 Tour'.",
        node=tour_name_leaf,
        additional_instruction="Look for explicit mention of the 2026 DWTS Live tour name in the answer."
    )

    # Tour Date Range Stated
    tour_range_leaf = evaluator.add_leaf(
        id="Tour_Date_Range_Stated",
        desc="State that the DWTS: Live! - 2026 Tour runs from January 22, 2026, to May 2, 2026.",
        parent=tour_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the tour runs from January 22, 2026 to May 2, 2026.",
        node=tour_range_leaf,
        additional_instruction="Accept minor formatting variations like 'Jan 22, 2026 – May 2, 2026'."
    )

    # Tour URL existence
    tour_url_exists = evaluator.add_custom_node(
        result=bool(data.tour and data.tour.urls and len(data.tour.urls) > 0),
        id="Tour_URL_Exists",
        desc="At least one official tour URL is provided.",
        parent=tour_group,
        critical=True,
    )

    # Tour URL support
    tour_url_leaf = evaluator.add_leaf(
        id="Tour_URL",
        desc="Provide a valid URL from an official source (DWTS tour site and/or official ticketing/venue page) confirming the tour and/or its date range.",
        parent=tour_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The provided official URL(s) confirm the 'Dancing With The Stars: Live! - 2026 Tour' and its schedule/date range.",
        node=tour_url_leaf,
        sources=(data.tour.urls if data.tour else []),
        additional_instruction="Accept only official sources such as the DWTS tour site, Ticketmaster event pages, or the venue's official site. The page should clearly indicate the 2026 tour and/or its schedule/date range."
    )

    # ----------------------- Venue and Date Information ----------------------- #
    venue_group = evaluator.add_parallel(
        id="Venue_and_Date_Information",
        desc="Confirm the venue is Radio City Music Hall in New York, NY, and the show date is Sunday, February 15, 2026, with supporting URL reference.",
        parent=main,
        critical=True,
    )

    # Venue Stated
    venue_leaf = evaluator.add_leaf(
        id="Venue_Stated",
        desc="State that the venue is Radio City Music Hall in New York, NY.",
        parent=venue_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the venue is Radio City Music Hall in New York, NY.",
        node=venue_leaf,
        additional_instruction="Allow minor variations like 'New York City, NY' or 'NYC'."
    )

    # Date Stated
    date_leaf = evaluator.add_leaf(
        id="Date_Stated",
        desc="State that the show date is Sunday, February 15, 2026.",
        parent=venue_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the show date is Sunday, February 15, 2026.",
        node=date_leaf,
        additional_instruction="Accept formats like 'Sun, Feb 15, 2026' or 'February 15, 2026 (Sunday)'."
    )

    # Venue/Date URL existence
    venue_date_url_exists = evaluator.add_custom_node(
        result=bool(data.venue_date and data.venue_date.urls and len(data.venue_date.urls) > 0),
        id="Venue_Date_URL_Exists",
        desc="At least one official venue/date URL is provided.",
        parent=venue_group,
        critical=True,
    )

    # Venue/Date URL support
    venue_date_url_leaf = evaluator.add_leaf(
        id="Venue_Date_URL",
        desc="Provide a valid URL from an official source (Ticketmaster, DWTS tour site, or venue website) confirming the venue and date.",
        parent=venue_group,
        critical=True,
    )
    venue_name = (data.venue_date.venue_name if data.venue_date else "Radio City Music Hall")
    city_state = (data.venue_date.city_state if data.venue_date else "New York, NY")
    show_date = (data.venue_date.show_date_text if data.venue_date else "Sunday, February 15, 2026")
    await evaluator.verify(
        claim=f"The official source confirms the show is scheduled at {venue_name} in {city_state} on {show_date}.",
        node=venue_date_url_leaf,
        sources=(data.venue_date.urls if data.venue_date else []),
        additional_instruction="Confirm both the venue (Radio City Music Hall) and the specific date (February 15, 2026) on the official page."
    )

    # --------------------- Performance Times Information ---------------------- #
    times_group = evaluator.add_parallel(
        id="Performance_Times_Information",
        desc="List both performance times scheduled for February 15, 2026, with an official supporting URL.",
        parent=main,
        critical=True,
    )

    # Matinee Time stated
    matinee_leaf = evaluator.add_leaf(
        id="Matinee_Time",
        desc="Specify the matinee performance time as 2:00 PM ET.",
        parent=times_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the matinee performance time is 2:00 PM ET.",
        node=matinee_leaf,
        additional_instruction="Allow reasonable formatting variants like '2:00 PM', '2 PM', or '2:00pm'; treat local New York time as ET."
    )

    # Evening Time stated
    evening_leaf = evaluator.add_leaf(
        id="Evening_Time",
        desc="Specify the evening performance time as 8:00 PM ET.",
        parent=times_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the evening performance time is 8:00 PM ET.",
        node=evening_leaf,
        additional_instruction="Allow reasonable formatting variants like '8:00 PM', '8 PM', or '8:00pm'; treat local New York time as ET."
    )

    # Performance Times URL existence
    perf_url_exists = evaluator.add_custom_node(
        result=bool(data.performance_times and data.performance_times.urls and len(data.performance_times.urls) > 0),
        id="Performance_Times_URL_Exists",
        desc="At least one official URL confirms the performance times.",
        parent=times_group,
        critical=True,
    )

    # Performance Times URL support
    perf_url_leaf = evaluator.add_leaf(
        id="Performance_Times_URL",
        desc="Provide a valid URL from an official source confirming both performance times for this date.",
        parent=times_group,
        critical=True,
    )
    matinee_time = (data.performance_times.matinee_time_text if data.performance_times else "2:00 PM")
    evening_time = (data.performance_times.evening_time_text if data.performance_times else "8:00 PM")
    await evaluator.verify(
        claim=f"The official source confirms that on February 15, 2026 there are two performances at {matinee_time} and {evening_time}.",
        node=perf_url_leaf,
        sources=(data.performance_times.urls if data.performance_times else []),
        additional_instruction="Confirm both times appear on the official page for the Feb 15, 2026 Radio City show."
    )

    # ----------------- Special Guest Co-Host Information ---------------------- #
    cohost_group = evaluator.add_parallel(
        id="Special_Guest_CoHost_Information",
        desc="Identify the Special Guest Co-Host scheduled for this date with supporting URL reference.",
        parent=main,
        critical=True,
    )

    # CoHost Name stated
    cohost_leaf = evaluator.add_leaf(
        id="CoHost_Name",
        desc="State that Danielle Fishel is the Special Guest Co-Host for this date (within her Jan 22 - Feb 15 appearance window).",
        parent=cohost_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that Danielle Fishel is the Special Guest Co-Host for the February 15, 2026 show.",
        node=cohost_leaf,
        additional_instruction="Minor name formatting variations are acceptable."
    )

    # CoHost URL existence
    cohost_url_exists = evaluator.add_custom_node(
        result=bool(data.cohost and data.cohost.urls and len(data.cohost.urls) > 0),
        id="CoHost_URL_Exists",
        desc="At least one official URL confirms the special guest co-host schedule.",
        parent=cohost_group,
        critical=True,
    )

    # CoHost URL support
    cohost_url_leaf = evaluator.add_leaf(
        id="CoHost_URL",
        desc="Provide a valid URL from an official source confirming the special guest co-host schedule.",
        parent=cohost_group,
        critical=True,
    )
    cohost_name = (data.cohost.name if data.cohost else "Danielle Fishel")
    await evaluator.verify(
        claim=f"The official source confirms that {cohost_name} is scheduled as the Special Guest Co-Host for the February 15, 2026 show (or appears within a date range that includes Feb 15, 2026).",
        node=cohost_url_leaf,
        sources=(data.cohost.urls if data.cohost else []),
        additional_instruction="Accept pages that list her appearance window (Jan 22–Feb 15) covering the date."
    )

    # ----------------------- VIP Package Information -------------------------- #
    vip_group = evaluator.add_parallel(
        id="VIP_Package_Information",
        desc="List all three VIP package tiers with required key features and provide supporting URL reference.",
        parent=main,
        critical=True,
    )

    # VIP Package Names stated
    vip_names_leaf = evaluator.add_leaf(
        id="VIP_Package_Names",
        desc="List all three VIP package tiers: Live From Hollywood Front Row VIP Package, Mirrorball Dreamin' VIP Package, and Judges' Choice VIP Package.",
        parent=vip_group,
        critical=True,
    )
    await evaluator.verify(
        claim=("The answer lists all three VIP package tiers: 'Live From Hollywood Front Row VIP Package', "
               "'Mirrorball Dreamin' VIP Package', and 'Judges' Choice VIP Package'."),
        node=vip_names_leaf,
        additional_instruction="Minor punctuation or capitalization variants are acceptable as long as the names clearly match."
    )

    # VIP Package Features stated
    vip_features_leaf = evaluator.add_leaf(
        id="VIP_Package_Features",
        desc="Include the required VIP features: interactive experience with DWTS Touring Cast members and professional photo opportunity.",
        parent=vip_group,
        critical=True,
    )
    await evaluator.verify(
        claim=("The answer includes the following VIP features: (1) an interactive experience with DWTS Touring Cast members, "
               "and (2) a professional photo opportunity."),
        node=vip_features_leaf,
        additional_instruction="Minor phrasing differences are acceptable if they clearly refer to these two features."
    )

    # VIP URL existence
    vip_url_exists = evaluator.add_custom_node(
        result=bool(data.vip and data.vip.urls and len(data.vip.urls) > 0),
        id="VIP_URL_Exists",
        desc="At least one official URL confirms VIP package details.",
        parent=vip_group,
        critical=True,
    )

    # VIP URL support
    vip_url_leaf = evaluator.add_leaf(
        id="VIP_URL",
        desc="Provide a valid URL from an official source confirming VIP package details.",
        parent=vip_group,
        critical=True,
    )
    await evaluator.verify(
        claim=("The official source confirms the VIP package tiers and that packages include an interactive experience "
               "with DWTS Touring Cast members and a professional photo opportunity."),
        node=vip_url_leaf,
        sources=(data.vip.urls if data.vip else []),
        additional_instruction="Prefer official DWTS tour pages, Ticketmaster VIP sections, or the venue's official VIP info."
    )

    # ------------------------ Ticketing Information --------------------------- #
    ticket_group = evaluator.add_parallel(
        id="Ticketing_Information",
        desc="Confirm the ticket purchasing platform and provide a direct purchase URL for this specific show.",
        parent=main,
        critical=True,
    )

    # Ticketing Platform stated
    platform_leaf = evaluator.add_leaf(
        id="Ticketing_Platform",
        desc="State that Ticketmaster is the official ticket purchasing platform.",
        parent=ticket_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that Ticketmaster is the official ticket purchasing platform.",
        node=platform_leaf,
        additional_instruction="Phrasing like 'tickets sold via Ticketmaster' is acceptable."
    )

    # Direct Purchase URL existence
    purchase_url_exists = evaluator.add_custom_node(
        result=bool(data.ticketing and data.ticketing.purchase_url and data.ticketing.purchase_url.strip()),
        id="Direct_Purchase_URL_Exists",
        desc="A direct Ticketmaster purchase URL for the February 15, 2026 Radio City show is provided.",
        parent=ticket_group,
        critical=True,
    )

    # Direct Purchase URL support
    purchase_url_leaf = evaluator.add_leaf(
        id="Direct_Purchase_URL",
        desc="Provide the direct Ticketmaster URL for purchasing tickets to this specific show on February 15, 2026.",
        parent=ticket_group,
        critical=True,
    )
    await evaluator.verify(
        claim=("This Ticketmaster URL is the direct purchase page for the Dancing With The Stars: Live! - 2026 Tour show at "
               "Radio City Music Hall in New York, NY on Sunday, February 15, 2026."),
        node=purchase_url_leaf,
        sources=(data.ticketing.purchase_url if data.ticketing else None),
        additional_instruction="Confirm the page is Ticketmaster and corresponds to Radio City Music Hall on Feb 15, 2026."
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
    Evaluate an answer for the DWTS Radio City Feb 15, 2026 task.
    """
    # Initialize evaluator (root is always non-critical; we add a critical top-level rubric node under it)
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_dwts_show(),
        template_class=DWTSShowExtraction,
        extraction_name="dwts_radiocity_2026_0215_extraction",
    )

    # Build verification tree and run checks
    await verify_dwts_show(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()