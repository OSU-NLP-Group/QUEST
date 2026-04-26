import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "summer2025_gaming_convention_highest_attendance"
TASK_DESCRIPTION = (
    "Among gaming conventions held in the United States during the summer months of 2025 (June through August), "
    "identify the convention that achieved the highest reported attendance. For this convention, provide the following "
    "information: (1) exact dates (start and end), (2) primary venue name(s), (3) city and state, and (4) the reported "
    "attendance figure. Conventions include video games, tabletop games, esports, or general gaming culture. The "
    "convention must have physically taken place in the U.S., and attendance must be from official or credible sources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConventionExtraction(BaseModel):
    """
    Extract exactly one selected convention (the one claimed to have the highest attendance among
    qualifying U.S. gaming conventions in summer 2025) and all required details, strictly from the answer.
    """
    # Selection/identification
    convention_name: Optional[str] = None
    selection_urls: List[str] = Field(default_factory=list)

    # Dates
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    dates_urls: List[str] = Field(default_factory=list)

    # Venues
    venues: List[str] = Field(default_factory=list)
    venue_urls: List[str] = Field(default_factory=list)

    # Location
    city: Optional[str] = None
    state: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    # Attendance
    attendance: Optional[str] = None
    attendance_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_convention() -> str:
    return """
    From the provided answer, extract details for exactly ONE gaming convention that the answer identifies as having
    the highest attendance among qualifying U.S. gaming conventions during Summer 2025 (June, July, or August 2025).

    If multiple conventions are mentioned, choose the one the answer explicitly claims has the highest attendance.
    If no such explicit claim is made, choose the first convention the answer ultimately presents as the final selection.

    Extract the following fields strictly from the answer as written (do NOT invent anything):
    - convention_name: The name/title of the selected convention.
    - selection_urls: An array of URLs cited in the answer that support the convention's eligibility and/or highest-attendance claim.

    - start_date: The exact start date as written (e.g., "August 2, 2025" or "Aug. 2, 2025").
    - end_date: The exact end date as written.
    - dates_urls: An array of URLs that support the dates.

    - venues: An array of the primary venue names (as written), e.g., ["Los Angeles Convention Center"].
    - venue_urls: An array of URLs that support the venue(s).

    - city: The city where the convention took place (as written).
    - state: The state where the convention took place (as written; either full name or abbreviation from the answer).
    - location_urls: An array of URLs that support the city/state.

    - attendance: The reported attendance figure (as written, e.g., "140,000", "140,000+", "~140k", "about 140,000").
    - attendance_urls: An array of URLs that support the attendance figure.

    Special rules for URLs:
    - Extract only URLs explicitly present in the answer text (including markdown links).
    - Do not infer or create URLs.
    - Include full URLs with protocol.

    If a field is missing from the answer, set it to null (for strings) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _conv_label(name: Optional[str]) -> str:
    return name if (name and name.strip()) else "the identified convention"


def _union_urls(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for urls in url_lists:
        if not urls:
            continue
        for u in urls:
            if isinstance(u, str):
                v = u.strip()
                if v and v not in seen:
                    seen.add(v)
                    merged.append(v)
    return merged


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_convention_selection(evaluator: Evaluator, parent_node, ext: ConventionExtraction) -> None:
    """
    Build and verify the 'Convention_Selection' subtree:
    - US_Location
    - Summer_2025_Timeframe
    - Gaming_Convention
    - Highest_Attendance
    - Selection_URLs (existence check)
    """
    selection_node = evaluator.add_parallel(
        id="Convention_Selection",
        desc="Correctly identify the gaming convention meeting all selection criteria",
        parent=parent_node,
        critical=True
    )

    # Selection_URLs existence (critical)
    evaluator.add_custom_node(
        result=bool(ext.selection_urls),
        id="Selection_URLs",
        desc="Reference URLs are provided that support the convention's eligibility and attendance ranking",
        parent=selection_node,
        critical=True
    )

    # Prepare shared info
    conv_name = _conv_label(ext.convention_name)
    all_selection_support = _union_urls(
        ext.selection_urls, ext.dates_urls, ext.location_urls, ext.venue_urls, ext.attendance_urls
    )

    # 1) US_Location (critical)
    us_loc_node = evaluator.add_leaf(
        id="US_Location",
        desc="The identified convention was held in the United States",
        parent=selection_node,
        critical=True
    )
    us_loc_claim = f"{conv_name} took place in the United States (USA)."
    # 2) Summer_2025_Timeframe (critical)
    timeframe_node = evaluator.add_leaf(
        id="Summer_2025_Timeframe",
        desc="The identified convention occurred during June, July, or August 2025",
        parent=selection_node,
        critical=True
    )
    timeframe_claim = (
        f"{conv_name} occurred during June, July, or August 2025 (the summer months of 2025)."
    )
    # 3) Gaming_Convention (critical)
    gaming_node = evaluator.add_leaf(
        id="Gaming_Convention",
        desc="The identified event is a gaming convention (video games, tabletop games, or general gaming culture)",
        parent=selection_node,
        critical=True
    )
    gaming_claim = (
        f"{conv_name} is a gaming convention focused on video games, tabletop games, esports, or general gaming culture."
    )
    # 4) Highest_Attendance (critical)
    highest_node = evaluator.add_leaf(
        id="Highest_Attendance",
        desc="The identified convention had the highest reported attendance among all qualifying U.S. gaming conventions during summer 2025",
        parent=selection_node,
        critical=True
    )
    att_phrase = f", with a reported attendance of {ext.attendance}" if (ext.attendance and ext.attendance.strip()) else ""
    highest_claim = (
        f"Among U.S. gaming conventions held in June–August 2025, {conv_name} had the highest reported attendance{att_phrase}."
    )

    # Batch verify the four critical checks using combined sources
    claims_and_sources = [
        (
            us_loc_claim,
            all_selection_support,
            us_loc_node,
            "Verify that the event physically took place in the United States. "
            "If multiple locations are listed, ensure they are U.S. locations."
        ),
        (
            timeframe_claim,
            _union_urls(ext.selection_urls, ext.dates_urls),
            timeframe_node,
            "Confirm the event's dates are within June 1 to August 31, 2025 (inclusive). "
            "If the page mentions month/year or a range spanning these months, that qualifies."
        ),
        (
            gaming_claim,
            _union_urls(ext.selection_urls, ext.venue_urls),
            gaming_node,
            "Confirm that the event is a gaming convention, i.e., focused on video games, tabletop games, "
            "esports, or general gaming culture. Consider official event descriptions."
        ),
        (
            highest_claim,
            _union_urls(ext.selection_urls, ext.attendance_urls),
            highest_node,
            "Determine if the sources explicitly support that this event had the highest reported attendance among "
            "all qualifying U.S. gaming conventions held in June–August 2025. Look for explicit comparative statements "
            "or sufficiently clear comparisons of attendance figures. If unclear or contradicted, mark incorrect."
        ),
    ]
    await evaluator.batch_verify(claims_and_sources)


async def verify_convention_details(evaluator: Evaluator, parent_node, ext: ConventionExtraction) -> None:
    """
    Build and verify the 'Convention_Details' subtree with four parallel groups:
    - Event_Dates
    - Venue_Names
    - Location_Details
    - Attendance_Figure
    Critical URL-support leaves will be failed if URLs are missing or values are missing.
    """
    details_node = evaluator.add_parallel(
        id="Convention_Details",
        desc="Provide accurate details about the identified convention",
        parent=parent_node,
        critical=False
    )

    conv_name = _conv_label(ext.convention_name)

    # ---------------- Event_Dates ----------------
    dates_main = evaluator.add_parallel(
        id="Event_Dates",
        desc="Provide the exact dates of the convention",
        parent=details_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(ext.start_date and ext.start_date.strip()),
        id="Start_Date",
        desc="The specific start date is provided",
        parent=dates_main,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(ext.end_date and ext.end_date.strip()),
        id="End_Date",
        desc="The specific end date is provided",
        parent=dates_main,
        critical=False
    )

    dates_urls_node = evaluator.add_leaf(
        id="Dates_URLs",
        desc="Reference URLs support the provided dates",
        parent=dates_main,
        critical=True
    )

    if ext.start_date and ext.end_date and ext.dates_urls:
        dates_claim = (
            f"{conv_name} took place from {ext.start_date} to {ext.end_date} in 2025."
        )
        await evaluator.verify(
            claim=dates_claim,
            node=dates_urls_node,
            sources=ext.dates_urls,
            additional_instruction=(
                "Verify the event ran from the stated start to end dates. "
                "Allow minor formatting variants (e.g., 'Aug.' vs 'August', en-dashes). "
                "If the dates differ by a day due to time zones, prefer the event's official page."
            )
        )
    else:
        dates_urls_node.score = 0.0
        dates_urls_node.status = "failed"

    # ---------------- Venue_Names ----------------
    venue_main = evaluator.add_parallel(
        id="Venue_Names",
        desc="Identify the primary venue(s) where the convention was held",
        parent=details_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(ext.venues),
        id="Primary_Venues",
        desc="The primary venue name(s) are provided",
        parent=venue_main,
        critical=False
    )

    venue_urls_node = evaluator.add_leaf(
        id="Venue_URLs",
        desc="Reference URLs support the venue name(s)",
        parent=venue_main,
        critical=True
    )

    if ext.venues and ext.venue_urls:
        venues_list_text = "; ".join(v for v in ext.venues if isinstance(v, str) and v.strip()) or "unknown venue(s)"
        venue_claim = f"The primary venue(s) for {conv_name} were: {venues_list_text}."
        await evaluator.verify(
            claim=venue_claim,
            node=venue_urls_node,
            sources=ext.venue_urls,
            additional_instruction="Confirm that the named venue(s) are indeed the main location(s) for the event. "
                                   "Allow for official or commonly used venue names and minor name variants."
        )
    else:
        venue_urls_node.score = 0.0
        venue_urls_node.status = "failed"

    # ---------------- Location_Details ----------------
    loc_main = evaluator.add_parallel(
        id="Location_Details",
        desc="Provide the city and state where the convention occurred",
        parent=details_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(ext.city and ext.city.strip()),
        id="City",
        desc="The city is provided",
        parent=loc_main,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(ext.state and ext.state.strip()),
        id="State",
        desc="The state is provided",
        parent=loc_main,
        critical=False
    )

    loc_urls_node = evaluator.add_leaf(
        id="Location_URLs",
        desc="Reference URLs support the city and state",
        parent=loc_main,
        critical=True
    )

    if ext.city and ext.state and ext.location_urls:
        loc_claim = f"{conv_name} took place in {ext.city}, {ext.state}, United States."
        await evaluator.verify(
            claim=loc_claim,
            node=loc_urls_node,
            sources=ext.location_urls,
            additional_instruction="Confirm the city and state for the event. "
                                   "Allow state abbreviations vs full names (e.g., 'CA' vs 'California')."
        )
    else:
        loc_urls_node.score = 0.0
        loc_urls_node.status = "failed"

    # ---------------- Attendance_Figure ----------------
    att_main = evaluator.add_parallel(
        id="Attendance_Figure",
        desc="Provide the reported attendance number",
        parent=details_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(ext.attendance and ext.attendance.strip()),
        id="Specific_Number",
        desc="A specific attendance figure is provided",
        parent=att_main,
        critical=False
    )

    att_urls_node = evaluator.add_leaf(
        id="Attendance_URLs",
        desc="Reference URLs support the attendance figure",
        parent=att_main,
        critical=True
    )

    if ext.attendance and ext.attendance_urls:
        att_claim = f"The reported attendance for {conv_name} was {ext.attendance}."
        await evaluator.verify(
            claim=att_claim,
            node=att_urls_node,
            sources=ext.attendance_urls,
            additional_instruction=(
                "Verify the attendance figure. Allow reasonable rounding, '+' signs, or approximations "
                "(e.g., 'about 140,000' vs '140,000'). If multiple years are present, ensure the number "
                "corresponds to 2025 unless the answer explicitly states otherwise."
            )
        )
    else:
        att_urls_node.score = 0.0
        att_urls_node.status = "failed"


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
    Evaluate an answer for the 'highest-attendance U.S. gaming convention during Summer 2025' task.
    """
    # Initialize evaluator with a sequential root so details are skipped if selection fails
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
        default_model=model,
    )

    # Optional: explicit task node to mirror rubric's "Task_Completion"
    task_node = evaluator.add_sequential(
        id="Task_Completion",
        desc="Complete identification of the highest-attendance U.S. gaming convention during summer 2025 and provide required details",
        parent=root,
        critical=False  # Must be non-critical to allow non-critical children in our framework
    )

    # Extract structured info
    extraction: ConventionExtraction = await evaluator.extract(
        prompt=prompt_extract_convention(),
        template_class=ConventionExtraction,
        extraction_name="convention_extraction",
    )

    # Build verification tree
    await verify_convention_selection(evaluator, task_node, extraction)
    await verify_convention_details(evaluator, task_node, extraction)

    # Return standard summary
    return evaluator.get_summary()