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
TASK_ID = "holiday_hours_2025_2026"
TASK_DESCRIPTION = (
    "You are planning last-minute holiday shopping during the 2025-2026 holiday season and need to know store hours for multiple retailers. "
    "Provide the following information with supporting reference URLs:\n\n"
    "1. What time does Walmart close on Christmas Eve (December 24, 2025)?\n"
    "2. Name a national pharmacy chain that is confirmed to be open on Christmas Day (December 25, 2025).\n"
    "3. What time does Home Depot open on Black Friday (November 28, 2025)?\n"
    "4. What time does Aldi close on Christmas Eve (December 24, 2025)?\n\n"
    "For each answer, include a reference URL from your research that supports the information."
)

# Optional ground-truth expectations (informational only; verification uses cited URLs)
EXPECTED_WALMART_CHRISTMAS_EVE_CLOSE = "6:00 PM"
EXPECTED_HOME_DEPOT_BLACK_FRIDAY_OPEN = "6:00 AM"
EXPECTED_ALDI_CHRISTMAS_EVE_CLOSE = "4:00 PM"
HOLIDAY_DATES = {
    "christmas_eve": "December 24, 2025",
    "christmas_day": "December 25, 2025",
    "black_friday": "November 28, 2025"
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WalmartInfo(BaseModel):
    close_time_christmas_eve: Optional[str] = None
    open_status_christmas_eve: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PharmacyInfo(BaseModel):
    chain_name: Optional[str] = None
    open_status_christmas_day: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HomeDepotInfo(BaseModel):
    open_time_black_friday: Optional[str] = None
    open_status_black_friday: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class AldiInfo(BaseModel):
    close_time_christmas_eve: Optional[str] = None
    reduced_hours_status_christmas_eve: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HolidayHoursExtraction(BaseModel):
    walmart: Optional[WalmartInfo] = None
    pharmacy: Optional[PharmacyInfo] = None
    home_depot: Optional[HomeDepotInfo] = None
    aldi: Optional[AldiInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_holiday_hours() -> str:
    return (
        "Extract the holiday store-hours information exactly as stated in the provided answer. "
        "Return a JSON object with four sections: walmart, pharmacy, home_depot, aldi.\n\n"
        "For each section, extract:\n"
        "- walmart:\n"
        "  • close_time_christmas_eve: The stated closing time for Walmart on Christmas Eve (December 24, 2025), exactly as written (e.g., '6 PM', '6:00 p.m.'). If not stated, return null.\n"
        "  • open_status_christmas_eve: Whether Walmart is stated as open on Christmas Eve 2025 (e.g., 'open', 'open with special hours', 'closed'). If not stated, return null.\n"
        "  • urls: All reference URLs provided in the answer that support Walmart's Christmas Eve hours. Extract actual URLs only. If none, return an empty array.\n"
        "- pharmacy:\n"
        "  • chain_name: The named national pharmacy chain (e.g., CVS, Walgreens, Rite Aid) that is stated to be open on Christmas Day 2025. If not stated, return null.\n"
        "  • open_status_christmas_day: The statement about being open on Christmas Day 2025 (e.g., 'open', 'select locations open'). If not stated, return null.\n"
        "  • urls: All reference URLs provided that support the pharmacy Christmas Day open status. If none, return an empty array.\n"
        "- home_depot:\n"
        "  • open_time_black_friday: The stated opening time for Home Depot on Black Friday (November 28, 2025), exactly as written. If not stated, return null.\n"
        "  • open_status_black_friday: Whether Home Depot is stated to be open on Black Friday 2025. If not stated, return null.\n"
        "  • urls: All reference URLs supporting Home Depot Black Friday hours. If none, return an empty array.\n"
        "- aldi:\n"
        "  • close_time_christmas_eve: The stated closing time for Aldi on Christmas Eve (December 24, 2025), exactly as written. If not stated, return null.\n"
        "  • reduced_hours_status_christmas_eve: The statement about shortened/reduced hours for Aldi on Christmas Eve 2025 (e.g., 'reduced hours'). If not stated, return null.\n"
        "  • urls: All reference URLs supporting Aldi Christmas Eve hours. If none, return an empty array.\n\n"
        "Rules:\n"
        "1) Extract only what is explicitly stated in the answer; do not invent details.\n"
        "2) URLs must be actual links present in the answer; include Markdown link targets.\n"
        "3) If a field is missing, return null; if no URLs, return an empty array.\n"
        "4) Preserve time formats exactly as stated in the answer."
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_walmart(evaluator: Evaluator, parent_node, info: Optional[WalmartInfo]) -> None:
    group = evaluator.add_parallel(
        id="Walmart_Christmas_Eve_Hours",
        desc="Accurate information about Walmart's closing time on Christmas Eve 2025",
        parent=parent_node,
        critical=False
    )

    urls = info.urls if info and info.urls else []

    # Reference URL existence (critical)
    evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="Walmart_Reference_URL",
        desc="Provides a reference URL supporting Walmart Christmas Eve hours",
        parent=group,
        critical=True
    )

    # Closing time claim (critical) — verify against provided URLs
    time_node = evaluator.add_leaf(
        id="Walmart_Christmas_Eve_Time",
        desc="States that Walmart closes at 6:00 PM on Christmas Eve (December 24, 2025)",
        parent=group,
        critical=True
    )
    close_time = info.close_time_christmas_eve if info and info.close_time_christmas_eve else ""
    walmart_time_claim = (
        f"Walmart closes at {close_time} on Christmas Eve ({HOLIDAY_DATES['christmas_eve']})."
        if close_time else
        f"Walmart closes at an explicitly stated time on Christmas Eve ({HOLIDAY_DATES['christmas_eve']})."
    )
    await evaluator.verify(
        claim=walmart_time_claim,
        node=time_node,
        sources=urls,
        additional_instruction=(
            "Check that the referenced URL(s) clearly state Walmart's closing time for December 24, 2025. "
            "Allow equivalent phrasing for time (e.g., '6 PM', '6:00 p.m.'). "
            "If the answer's time does not match the URL evidence or the date is wrong, mark as incorrect."
        ),
    )

    # Open status (critical) — verify against URLs
    open_node = evaluator.add_leaf(
        id="Walmart_Open_Status",
        desc="Confirms Walmart is open on Christmas Eve 2025 (not closed all day)",
        parent=group,
        critical=True
    )
    open_claim = f"Walmart is open on Christmas Eve ({HOLIDAY_DATES['christmas_eve']})."
    await evaluator.verify(
        claim=open_claim,
        node=open_node,
        sources=urls,
        additional_instruction=(
            "Verify the URL(s) indicate Walmart operates on December 24, 2025 (not closed all day). "
            "Accept statements like 'reduced hours' or 'holiday hours' indicating open status."
        ),
    )


async def verify_pharmacy(evaluator: Evaluator, parent_node, info: Optional[PharmacyInfo]) -> None:
    group = evaluator.add_parallel(
        id="Pharmacy_Christmas_Day",
        desc="Identifies a national pharmacy chain open on Christmas Day 2025",
        parent=parent_node,
        critical=False
    )

    urls = info.urls if info and info.urls else []
    chain_name = info.chain_name if info and info.chain_name else ""

    # Reference URL existence (critical)
    evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="Pharmacy_Reference_URL",
        desc="Provides a reference URL supporting pharmacy Christmas Day hours",
        parent=group,
        critical=True
    )

    # Chain name (critical) — verify the answer names a national pharmacy chain
    chain_node = evaluator.add_leaf(
        id="Pharmacy_Chain_Name",
        desc="Names a national pharmacy chain (e.g., CVS, Walgreens, or similar)",
        parent=group,
        critical=True
    )
    name_claim = (
        f"The answer names the national pharmacy chain '{chain_name}'."
        if chain_name else
        "The answer names at least one national pharmacy chain (e.g., CVS, Walgreens, Rite Aid)."
    )
    await evaluator.verify(
        claim=name_claim,
        node=chain_node,
        additional_instruction=(
            "Judge based on the answer text only. Consider well-known national chains like CVS, Walgreens, or Rite Aid "
            "as valid examples. If no chain name appears, mark incorrect."
        ),
    )

    # Christmas Day open status (critical) — verify against URLs
    open_node = evaluator.add_leaf(
        id="Pharmacy_Christmas_Open",
        desc="Confirms the pharmacy is open on Christmas Day (December 25, 2025)",
        parent=group,
        critical=True
    )
    open_claim = (
        f"{chain_name} is open on Christmas Day ({HOLIDAY_DATES['christmas_day']})."
        if chain_name else
        f"A named national pharmacy chain is open on Christmas Day ({HOLIDAY_DATES['christmas_day']})."
    )
    await evaluator.verify(
        claim=open_claim,
        node=open_node,
        sources=urls,
        additional_instruction=(
            "Confirm that the referenced URL(s) indicate the pharmacy chain operates on December 25, 2025. "
            "Accept phrasing like 'select locations open' as open status."
        ),
    )


async def verify_home_depot(evaluator: Evaluator, parent_node, info: Optional[HomeDepotInfo]) -> None:
    group = evaluator.add_parallel(
        id="Home_Depot_Black_Friday_Hours",
        desc="Accurate information about Home Depot's opening time on Black Friday 2025",
        parent=parent_node,
        critical=False
    )

    urls = info.urls if info and info.urls else []
    open_time = info.open_time_black_friday if info and info.open_time_black_friday else ""

    # Reference URL existence (critical)
    evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="Home_Depot_Reference_URL",
        desc="Provides a reference URL supporting Home Depot Black Friday hours",
        parent=group,
        critical=True
    )

    # Opening time claim (critical) — verify against URLs
    time_node = evaluator.add_leaf(
        id="Home_Depot_Opening_Time",
        desc="States that Home Depot opens at 6:00 AM on Black Friday (November 28, 2025)",
        parent=group,
        critical=True
    )
    time_claim = (
        f"Home Depot opens at {open_time} on Black Friday ({HOLIDAY_DATES['black_friday']})."
        if open_time else
        f"Home Depot opens at an explicitly stated time on Black Friday ({HOLIDAY_DATES['black_friday']})."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_node,
        sources=urls,
        additional_instruction=(
            "Check that the referenced URL(s) explicitly state Home Depot's Black Friday opening time for November 28, 2025. "
            "Allow equivalent phrasing for time (e.g., '6 AM', '6:00 a.m.'). "
            "If the answer's time does not match the URL evidence or the date is wrong, mark as incorrect."
        ),
    )

    # Open status (critical) — verify against URLs
    open_node = evaluator.add_leaf(
        id="Home_Depot_Open_Status",
        desc="Confirms Home Depot is open on Black Friday 2025",
        parent=group,
        critical=True
    )
    open_claim = f"Home Depot is open on Black Friday ({HOLIDAY_DATES['black_friday']})."
    await evaluator.verify(
        claim=open_claim,
        node=open_node,
        sources=urls,
        additional_instruction=(
            "Verify the URL(s) indicate Home Depot operates on November 28, 2025 (Black Friday). "
            "If any URL indicates closure or contradicts 2025, mark incorrect."
        ),
    )


async def verify_aldi(evaluator: Evaluator, parent_node, info: Optional[AldiInfo]) -> None:
    group = evaluator.add_parallel(
        id="Aldi_Christmas_Eve_Hours",
        desc="Accurate information about Aldi's closing time on Christmas Eve 2025",
        parent=parent_node,
        critical=False
    )

    urls = info.urls if info and info.urls else []
    close_time = info.close_time_christmas_eve if info and info.close_time_christmas_eve else ""

    # Reference URL existence (critical)
    evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="Aldi_Reference_URL",
        desc="Provides a reference URL supporting Aldi Christmas Eve hours",
        parent=group,
        critical=True
    )

    # Closing time claim (critical) — verify against URLs
    time_node = evaluator.add_leaf(
        id="Aldi_Christmas_Eve_Time",
        desc="States that Aldi closes at 4:00 PM on Christmas Eve (December 24, 2025)",
        parent=group,
        critical=True
    )
    time_claim = (
        f"Aldi closes at {close_time} on Christmas Eve ({HOLIDAY_DATES['christmas_eve']})."
        if close_time else
        f"Aldi closes at an explicitly stated time on Christmas Eve ({HOLIDAY_DATES['christmas_eve']})."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_node,
        sources=urls,
        additional_instruction=(
            "Check that the referenced URL(s) clearly state Aldi's closing time for December 24, 2025. "
            "Allow equivalent phrasing for time (e.g., '4 PM', '4:00 p.m.'). "
            "If the answer's time does not match the URL evidence or the date is wrong, mark as incorrect."
        ),
    )

    # Reduced hours status (critical) — verify against URLs
    reduced_node = evaluator.add_leaf(
        id="Aldi_Reduced_Hours",
        desc="Confirms Aldi operates with shortened/reduced hours on Christmas Eve 2025",
        parent=group,
        critical=True
    )
    reduced_claim = f"Aldi operates with shortened or reduced hours on Christmas Eve ({HOLIDAY_DATES['christmas_eve']})."
    await evaluator.verify(
        claim=reduced_claim,
        node=reduced_node,
        sources=urls,
        additional_instruction=(
            "Verify the URL(s) indicate Aldi has shortened/reduced hours on December 24, 2025. "
            "Statements like 'holiday hours' or 'limited hours' count as reduced hours."
        ),
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
    Evaluate an answer for the holiday shopping hours task (2025-2026).
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Allow independent checks for each retailer
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

    # Root node description (set explicitly for clarity, keep non-critical to allow partial credit)
    root.desc = "Complete and accurate information about store hours during the 2025-2026 holiday shopping season"

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_holiday_hours(),
        template_class=HolidayHoursExtraction,
        extraction_name="holiday_hours_extraction",
    )

    # Add Ground Truth info (informational only)
    evaluator.add_ground_truth({
        "expected_walmart_christmas_eve_close": EXPECTED_WALMART_CHRISTMAS_EVE_CLOSE,
        "expected_home_depot_black_friday_open": EXPECTED_HOME_DEPOT_BLACK_FRIDAY_OPEN,
        "expected_aldi_christmas_eve_close": EXPECTED_ALDI_CHRISTMAS_EVE_CLOSE,
        "holiday_dates": HOLIDAY_DATES
    }, gt_type="expected_values")

    # Build subtrees and run verifications
    await asyncio.gather(
        verify_walmart(evaluator, root, extracted.walmart),
        verify_pharmacy(evaluator, root, extracted.pharmacy),
        verify_home_depot(evaluator, root, extracted.home_depot),
        verify_aldi(evaluator, root, extracted.aldi),
    )

    # Return evaluator summary including verification tree and scores
    return evaluator.get_summary()