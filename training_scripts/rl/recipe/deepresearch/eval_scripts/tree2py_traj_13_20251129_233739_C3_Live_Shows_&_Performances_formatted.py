import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broadway_tour_nys_2025"
TASK_DESCRIPTION = (
    "I'm planning to visit New York State during the late fall/winter holiday season of 2025 and would like to see a touring Broadway show. "
    "Find one Broadway touring production that has an engagement in New York State starting between November 1 and December 31, 2025, with a run of at least 10 consecutive days. "
    "Provide the show title, the specific city and venue in New York State where it will perform, and the exact start and end dates of that engagement."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TourEngagement(BaseModel):
    """
    One selected Broadway touring engagement as provided in the answer.
    """
    show_title: Optional[str] = None
    city: Optional[str] = None
    venue: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_engagement() -> str:
    return """
    Extract the single Broadway touring engagement presented in the answer that is intended to meet the user's constraints.
    Return the following fields:
    - show_title: The title of the touring Broadway show.
    - city: The specific city in New York State where the engagement occurs (do not include state/abbreviation here; city name only as written in the answer).
    - venue: The venue name exactly as written in the answer (ideally matching the official tour schedule or the venue's own event listing).
    - start_date: The exact start date of the engagement as written in the answer (keep the original format, do not normalize).
    - end_date: The exact end date of the engagement as written in the answer (keep the original format, do not normalize).
    - sources: A list of URLs explicitly cited in the answer that directly support this engagement (e.g., official tour schedule, the venue's event page, ticketing page about this run). Only include URLs that appear in the answer; do not invent or infer any new URLs.

    RULES:
    - Extract only what is explicitly present in the answer text.
    - If any field is missing, set it to null (for strings) or [] (for the sources list).
    - Do not infer or fabricate dates, venues, or locations that are not written in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nz(s: Optional[str]) -> str:
    """Return non-empty string or empty string."""
    return s or ""


def _urls(urls: Optional[List[str]]) -> List[str]:
    """Normalize URL list."""
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_rubric(
    evaluator: Evaluator,
    engagement: TourEngagement,
) -> None:
    """
    Build the rubric tree per the provided JSON and run verifications.
    """

    # Root (container) already exists; create a critical main task node under it
    main_node = evaluator.add_parallel(
        id="Broadway_Tour_Task",
        desc="Identify one Broadway touring production with a qualifying New York State engagement starting in late 2025 and provide the required engagement details.",
        parent=evaluator.root,
        critical=True,
    )

    # Prepare fields
    title = _nz(engagement.show_title)
    city = _nz(engagement.city)
    venue = _nz(engagement.venue)
    start_date = _nz(engagement.start_date)
    end_date = _nz(engagement.end_date)
    sources = _urls(engagement.sources)

    # ------------------------------------------------------------------ #
    # 1) Show_Is_Broadway_Touring_Production (critical leaf)             #
    # ------------------------------------------------------------------ #
    touring_leaf = evaluator.add_leaf(
        id="Show_Is_Broadway_Touring_Production",
        desc="The selected show is a Broadway touring production (not resident-only).",
        parent=main_node,
        critical=True,
    )
    touring_claim = (
        f"The show '{title}' is an official Broadway touring production (i.e., a national tour or Broadway tour), "
        f"and the cited sources indicate it is a touring production rather than a resident-only production."
    )
    await evaluator.verify(
        claim=touring_claim,
        node=touring_leaf,
        sources=sources,
        additional_instruction=(
            "Use only the information from the cited sources (tour schedule pages, official show pages, venue/event pages) "
            "to confirm that this engagement is part of a touring Broadway production. Look for clear indicators like "
            "'National Tour', 'Broadway tour', 'North American tour', or inclusion on an official tour schedule. "
            "Do not rely on your own memory."
        ),
    )

    # ------------------------------------------------------------------ #
    # 2) Engagement_Meets_Time_And_Location_Constraints (critical agg)   #
    # ------------------------------------------------------------------ #
    time_loc_node = evaluator.add_parallel(
        id="Engagement_Meets_Time_And_Location_Constraints",
        desc="The provided engagement is in New York State and satisfies the specified start-date window and minimum run length.",
        parent=main_node,
        critical=True,
    )

    # 2.a) Engagement_In_New_York_State (critical leaf)
    nys_leaf = evaluator.add_leaf(
        id="Engagement_In_New_York_State",
        desc="The engagement location (city/venue) is within New York State.",
        parent=time_loc_node,
        critical=True,
    )
    nys_claim = (
        f"The engagement for '{title}' at '{venue}' in '{city}' is located in New York State (NY). "
        f"At least one cited source page explicitly shows this event/location in NY."
    )
    await evaluator.verify(
        claim=nys_claim,
        node=nys_leaf,
        sources=sources,
        additional_instruction=(
            "Check the source(s) for explicit location text such as 'NY', 'New York', or the city being shown with 'NY' on the event or tour page. "
            "Do not rely on your own knowledge; the evidence must be visible in the webpage text or screenshot."
        ),
    )

    # 2.b) Engagement_Start_Between_Nov1_And_Dec31_2025 (critical leaf)
    window_leaf = evaluator.add_leaf(
        id="Engagement_Start_Between_Nov1_And_Dec31_2025",
        desc="The engagement start date is between November 1, 2025 and December 31, 2025 (inclusive).",
        parent=time_loc_node,
        critical=True,
    )
    window_claim = (
        f"The engagement starts on {start_date}, and that start date falls between November 1, 2025 and December 31, 2025 inclusive."
    )
    await evaluator.verify(
        claim=window_claim,
        node=window_leaf,
        sources=sources,
        additional_instruction=(
            "Use the dates shown on the cited sources to verify the start date and check that it is within the inclusive range "
            "11/01/2025 — 12/31/2025. If the exact date is visible in an image but not the text, consult the screenshot."
        ),
    )

    # 2.c) Engagement_Run_At_Least_10_Consecutive_Days (critical leaf)
    runlen_leaf = evaluator.add_leaf(
        id="Engagement_Run_At_Least_10_Consecutive_Days",
        desc="The engagement lasts at least 10 consecutive days (end date - start date + 1 >= 10).",
        parent=time_loc_node,
        critical=True,
    )
    runlen_claim = (
        f"The engagement for '{title}' runs from {start_date} to {end_date}, inclusive, and the run length is at least "
        f"10 consecutive days (computed as end_date - start_date + 1 >= 10) based on the cited source(s)."
    )
    await evaluator.verify(
        claim=runlen_claim,
        node=runlen_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm both the start and end dates from the source(s) and compute the inclusive day count. "
            "If the interval spans multiple months, ensure the calculation is correct across month boundaries."
        ),
    )

    # ------------------------------------------------------------------ #
    # 3) Venue_Qualification (critical agg)                              #
    # ------------------------------------------------------------------ #
    venue_main = evaluator.add_parallel(
        id="Venue_Qualification",
        desc="Venue satisfies venue-type constraint.",
        parent=main_node,
        critical=True,
    )

    venue_host_leaf = evaluator.add_leaf(
        id="Venue_Is_Professional_Touring_Host",
        desc="The venue is a professional theater that hosts touring Broadway productions.",
        parent=venue_main,
        critical=True,
    )
    venue_host_claim = (
        f"The venue '{venue}' is a professional theater or performing arts center that hosts touring Broadway productions, "
        f"as evidenced by the cited sources (e.g., part of a Broadway series or regularly presents national tours)."
    )
    await evaluator.verify(
        claim=venue_host_claim,
        node=venue_host_leaf,
        sources=sources,
        additional_instruction=(
            "Look for indications that the venue presents touring Broadway shows (e.g., 'Broadway Series', 'Broadway Across America', "
            "'national tour engagements' listed). Avoid relying on general knowledge; confirm via the provided source(s)."
        ),
    )

    # ------------------------------------------------------------------ #
    # 4) Required_Output_Provided (critical agg)                         #
    # ------------------------------------------------------------------ #
    req_out = evaluator.add_parallel(
        id="Required_Output_Provided",
        desc="All required output fields are provided, including exact venue naming and engagement dates.",
        parent=main_node,
        critical=True,
    )

    # 4.a) Show_Title_Provided (critical existence check)
    evaluator.add_custom_node(
        result=bool(title.strip()),
        id="Show_Title_Provided",
        desc="Show title is provided.",
        parent=req_out,
        critical=True,
    )

    # 4.b) City_Provided (critical existence check)
    evaluator.add_custom_node(
        result=bool(city.strip()),
        id="City_Provided",
        desc="Specific city for the New York State engagement is provided.",
        parent=req_out,
        critical=True,
    )

    # 4.c) Venue_Name_Provided_Exactly_As_Official_Schedule (critical, verified against source)
    venue_exact_leaf = evaluator.add_leaf(
        id="Venue_Name_Provided_Exactly_As_Official_Schedule",
        desc="Venue name is provided exactly as listed on the official tour schedule.",
        parent=req_out,
        critical=True,
    )
    venue_exact_claim = (
        f"On at least one of the cited source pages, the venue name for this specific engagement appears exactly as '{venue}'."
    )
    await evaluator.verify(
        claim=venue_exact_claim,
        node=venue_exact_leaf,
        sources=sources,
        additional_instruction=(
            "Check the official tour schedule or the venue's event listing among the cited sources. "
            "The string should match exactly aside from trivial whitespace and letter casing. "
            "If the provided venue name is truncated, expanded, or otherwise different from how it appears in the source, mark this as not supported."
        ),
    )

    # 4.d) Exact_Start_And_End_Dates_Provided (critical existence check)
    evaluator.add_custom_node(
        result=bool(start_date.strip()) and bool(end_date.strip()),
        id="Exact_Start_And_End_Dates_Provided",
        desc="Exact engagement start date and exact engagement end date are both provided.",
        parent=req_out,
        critical=True,
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
    Evaluate an answer for the Broadway touring NYS 2025 engagement task.
    """

    # Initialize evaluator and root
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
        default_model=model,
    )

    # Record constraints as info (optional)
    evaluator.add_custom_info(
        {
            "state": "New York",
            "start_window_inclusive": ["2025-11-01", "2025-12-31"],
            "min_consecutive_days": 10
        },
        info_type="constraints",
        info_name="task_constraints"
    )

    # Extract the engagement info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_engagement(),
        template_class=TourEngagement,
        extraction_name="selected_engagement"
    )

    # Build verification tree and run checks
    await build_and_verify_rubric(evaluator, extracted)

    # Return structured summary
    return evaluator.get_summary()