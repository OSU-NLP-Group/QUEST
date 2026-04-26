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
TASK_ID = "thanksgiving_pa_2025_plan"
TASK_DESCRIPTION = (
    "You are planning your Thanksgiving weekend activities in Pennsylvania for 2025 and hold a Sam's Club Plus membership. "
    "You need to plan your shopping schedule across three days (Thanksgiving Eve, Thanksgiving Day, and Black Friday). "
    "Based on the 2025 holiday schedules and promotional offers, provide the following information: "
    "(1) Will Wegmans grocery stores be open on Thanksgiving Day (November 27, 2025) in Pennsylvania? If yes, what time will they close? "
    "(2) What are the complete details of Chipotle's Thanksgiving Eve promotion on November 26, 2025? Specifically provide: the promotion name, "
    "the exact time window when it is valid, where/how it can be redeemed, and any transaction limits that apply. "
    "(3) What are the Black Friday (November 28, 2025) operating hours for Sam's Club, specifically noting: what time Sam's Club Plus members can begin shopping, "
    "what time regular club members can begin shopping, and what time the store closes? "
    "(4) Calculate the total number of hours you will have available to shop at Sam's Club on Black Friday (November 28, 2025) with your Plus membership early access, "
    "from when you can first enter until the store closes. For each piece of information, provide supporting reference URLs from official sources or reliable news outlets."
)

# Expected values used for verification guidance
EXPECTED_CHIPOTLE_PROMO_NAME = "Back Home BOGO"
EXPECTED_SAMS_PLUS_START = "8:00 AM"
EXPECTED_SAMS_REGULAR_START = "9:00 AM"
EXPECTED_SAMS_CLOSE = "8:00 PM"
EXPECTED_TOTAL_HOURS = "12"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WegmansInfo(BaseModel):
    open_status: Optional[str] = None  # e.g., "open", "closed"
    closing_time: Optional[str] = None  # e.g., "4 PM", null if closed
    sources: List[str] = Field(default_factory=list)


class ChipotlePromo(BaseModel):
    promotion_name: Optional[str] = None
    time_window: Optional[str] = None
    redemption_channel: Optional[str] = None
    transaction_limit: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SamsClubHours(BaseModel):
    plus_member_start: Optional[str] = None
    regular_member_start: Optional[str] = None
    store_close: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TotalDurationInfo(BaseModel):
    hours_value: Optional[str] = None  # Keep as string (e.g., "12", "12 hours", "twelve")
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_wegmans() -> str:
    return (
        "Extract the answer's information about Wegmans grocery stores in Pennsylvania for Thanksgiving Day (Nov 27, 2025).\n"
        "Return a JSON with:\n"
        "- open_status: 'open' or 'closed' exactly as stated in the answer (or a synonymous phrasing; choose the closest canonical word).\n"
        "- closing_time: If the answer states Wegmans is open, provide the stated closing time string exactly as in the answer (e.g., '4 PM', '4:00 p.m.'); otherwise return null.\n"
        "- sources: An array of all supporting URLs explicitly mentioned in the answer that substantiate the Wegmans Thanksgiving Day open/close status and (if applicable) closing time. "
        "Extract only URLs explicitly present in the answer text (including markdown links)."
    )


def prompt_extract_chipotle() -> str:
    return (
        "Extract Chipotle's Thanksgiving Eve promotion details for Nov 26, 2025 as presented in the answer.\n"
        "Return a JSON with:\n"
        "- promotion_name: The promotion name exactly as stated in the answer.\n"
        "- time_window: The validity window string exactly as in the answer (e.g., 'Nov 26, 2025 from 4:00 PM until close (local time)').\n"
        "- redemption_channel: Where/how it can be redeemed exactly as stated (e.g., 'in-restaurant only; not valid for catering, mobile, online, or delivery').\n"
        "- transaction_limit: The transaction limit text (e.g., 'limited to 5 free entrées per check/transaction').\n"
        "- sources: An array of all supporting URLs explicitly mentioned in the answer for these promotion details (official sources or reliable news)."
    )


def prompt_extract_sams() -> str:
    return (
        "Extract Sam's Club Black Friday (Nov 28, 2025) operating hours details from the answer.\n"
        "Return a JSON with:\n"
        "- plus_member_start: The stated start time for Sam's Club Plus members (e.g., '8:00 AM').\n"
        "- regular_member_start: The stated start time for regular club members (e.g., '9:00 AM').\n"
        "- store_close: The stated closing time (e.g., '8:00 PM').\n"
        "- sources: An array of all supporting URLs explicitly mentioned in the answer for these hours."
    )


def prompt_extract_total_duration() -> str:
    return (
        "Extract the total number of hours the answer states are available to shop at Sam's Club on Black Friday (Nov 28, 2025) with Plus membership early access, "
        "from first entry until close.\n"
        "Return a JSON with:\n"
        "- hours_value: The value exactly as stated (e.g., '12', '12 hours', 'twelve hours').\n"
        "- sources: An array of supporting URLs the answer cites for the underlying hours used in this calculation "
        "(the answer may reuse the Sam's Club hours sources). Extract only URLs explicitly present in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalized_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _is_open_status(status: Optional[str]) -> bool:
    st = _normalized_text(status)
    return "open" in st and "closed" not in st


def _has_nonempty(val: Optional[str]) -> bool:
    return bool(val and val.strip())


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_wegmans_verification(evaluator: Evaluator, parent_node, weg: WegmansInfo) -> None:
    q_node = evaluator.add_parallel(
        id="Question_1_Wegmans_Thanksgiving_Day_PA",
        desc="Answers whether Wegmans in Pennsylvania is open on Thanksgiving Day (Nov 27, 2025) and, if open, provides the closing time, with supporting URLs.",
        parent=parent_node,
        critical=False,
    )

    # Leaf: Open or closed status (critical)
    status_leaf = evaluator.add_leaf(
        id="Wegmans_Open_or_Closed_Status_Provided",
        desc="States whether Wegmans stores in Pennsylvania are open or closed on Thanksgiving Day (Nov 27, 2025).",
        parent=q_node,
        critical=True,
    )
    status_text = weg.open_status or ""
    status_claim = (
        f"Wegmans grocery stores in Pennsylvania will be {status_text} on Thanksgiving Day (November 27, 2025)."
        if status_text else
        "The answer states whether Wegmans grocery stores in Pennsylvania are open or closed on Thanksgiving Day (November 27, 2025)."
    )
    await evaluator.verify(
        claim=status_claim,
        node=status_leaf,
        sources=weg.sources,
        additional_instruction="Verify the stated open/closed status for Wegmans on Thanksgiving Day in Pennsylvania using the provided URLs."
    )

    # Custom: Closing time presence conditional on 'open'
    evaluator.add_custom_node(
        result=(not _is_open_status(weg.open_status)) or _has_nonempty(weg.closing_time),
        id="Wegmans_Closing_Time_If_Open",
        desc="If the answer states Wegmans is open, it provides a Thanksgiving Day closing time for Pennsylvania locations (a specific time is stated).",
        parent=q_node,
        critical=True
    )

    # Leaf: Reference URLs substantiate the status and closing time if applicable (critical)
    refs_leaf = evaluator.add_leaf(
        id="Wegmans_Reference_URLs",
        desc="Provides supporting reference URL(s) from official sources or reliable news outlets that substantiate the Wegmans Thanksgiving Day open/close status and (if applicable) the stated closing time.",
        parent=q_node,
        critical=True
    )
    if _is_open_status(weg.open_status):
        refs_claim = (
            f"The provided sources substantiate that Wegmans stores in Pennsylvania are open on Thanksgiving Day "
            f"(Nov 27, 2025) and that they close at {weg.closing_time} (local time)."
        )
    else:
        refs_claim = (
            "The provided sources substantiate that Wegmans stores in Pennsylvania are closed on Thanksgiving Day (Nov 27, 2025)."
        )
    await evaluator.verify(
        claim=refs_claim,
        node=refs_leaf,
        sources=weg.sources,
        additional_instruction="Assess whether the URLs explicitly confirm the Wegmans Thanksgiving Day status and, if open, the specific closing time for Pennsylvania."
    )


async def build_chipotle_verification(evaluator: Evaluator, parent_node, chip: ChipotlePromo) -> None:
    q_node = evaluator.add_parallel(
        id="Question_2_Chipotle_Thanksgiving_Eve_Promotion",
        desc="Provides complete details of Chipotle's Thanksgiving Eve promotion on Nov 26, 2025, with supporting URLs.",
        parent=parent_node,
        critical=False,
    )

    # Promotion name
    name_leaf = evaluator.add_leaf(
        id="Chipotle_Promotion_Name",
        desc="Identifies the promotion name as 'Back Home BOGO' (BOGO).",
        parent=q_node,
        critical=True,
    )
    name_claim = (
        "The Chipotle Thanksgiving Eve promotion on November 26, 2025 is named 'Back Home BOGO'."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=chip.sources,
        additional_instruction="Confirm that the promotion name is 'Back Home BOGO' as stated by Chipotle or reliable coverage."
    )

    # Time window
    time_leaf = evaluator.add_leaf(
        id="Chipotle_Time_Window",
        desc="States the validity window as Nov 26, 2025 from 4:00 PM until close (local time).",
        parent=q_node,
        critical=True,
    )
    time_claim = (
        "The Chipotle Thanksgiving Eve promotion is valid on November 26, 2025 from 4:00 PM until close (local time)."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        sources=chip.sources,
        additional_instruction="Verify the promotion's time window including the date, 4:00 PM start, and 'until close' local time."
    )

    # Redemption channel
    redeem_leaf = evaluator.add_leaf(
        id="Chipotle_Redemption_Channel",
        desc="States where/how it can be redeemed: in-restaurant only (and not valid for catering, mobile, online, or delivery).",
        parent=q_node,
        critical=True,
    )
    redeem_claim = (
        "The Chipotle promotion is redeemable in-restaurant only and is not valid for catering, mobile, online, or delivery."
    )
    await evaluator.verify(
        claim=redeem_claim,
        node=redeem_leaf,
        sources=chip.sources,
        additional_instruction="Confirm redemption restrictions: in-restaurant only; excludes catering, mobile, online, and delivery."
    )

    # Transaction limit
    limit_leaf = evaluator.add_leaf(
        id="Chipotle_Transaction_Limit",
        desc="States the transaction limit: limited to 5 free entrées per check/transaction.",
        parent=q_node,
        critical=True,
    )
    limit_claim = "The Chipotle promotion is limited to 5 free entrées per check/transaction."
    await evaluator.verify(
        claim=limit_claim,
        node=limit_leaf,
        sources=chip.sources,
        additional_instruction="Verify the transaction limit is 5 free entrées per check/transaction."
    )

    # Reference URLs existence
    evaluator.add_custom_node(
        result=bool(chip.sources),
        id="Chipotle_Reference_URLs",
        desc="Provides supporting reference URL(s) from official sources or reliable news outlets for the Chipotle promotion details.",
        parent=q_node,
        critical=True
    )


async def build_sams_verification(evaluator: Evaluator, parent_node, sams: SamsClubHours) -> None:
    q_node = evaluator.add_parallel(
        id="Question_3_Sams_Club_Black_Friday_Hours",
        desc="Provides Sam's Club Black Friday (Nov 28, 2025) operating hours details requested, with supporting URLs.",
        parent=parent_node,
        critical=False,
    )

    # Plus member start time
    plus_leaf = evaluator.add_leaf(
        id="Plus_Member_Start_Time",
        desc="States Plus members can begin shopping at 8:00 AM.",
        parent=q_node,
        critical=True,
    )
    plus_claim = "On Black Friday (November 28, 2025), Sam's Club Plus members can begin shopping at 8:00 AM."
    await evaluator.verify(
        claim=plus_claim,
        node=plus_leaf,
        sources=sams.sources,
        additional_instruction="Confirm that Plus members' early access starts at 8:00 AM on Black Friday 2025."
    )

    # Regular member start time
    regular_leaf = evaluator.add_leaf(
        id="Regular_Member_Start_Time",
        desc="States regular club members can begin shopping at 9:00 AM.",
        parent=q_node,
        critical=True,
    )
    regular_claim = "On Black Friday (November 28, 2025), regular Sam's Club members can begin shopping at 9:00 AM."
    await evaluator.verify(
        claim=regular_claim,
        node=regular_leaf,
        sources=sams.sources,
        additional_instruction="Confirm that regular members' start time is 9:00 AM on Black Friday 2025."
    )

    # Store closing time
    close_leaf = evaluator.add_leaf(
        id="Store_Closing_Time",
        desc="States the store closes at 8:00 PM.",
        parent=q_node,
        critical=True,
    )
    close_claim = "On Black Friday (November 28, 2025), Sam's Club stores close at 8:00 PM."
    await evaluator.verify(
        claim=close_claim,
        node=close_leaf,
        sources=sams.sources,
        additional_instruction="Confirm the Black Friday store closing time is 8:00 PM."
    )

    # Reference URLs existence
    evaluator.add_custom_node(
        result=bool(sams.sources),
        id="Sams_Club_Reference_URLs",
        desc="Provides supporting reference URL(s) from official sources or reliable news outlets for the Sam's Club Black Friday hours.",
        parent=q_node,
        critical=True
    )


async def build_duration_verification(evaluator: Evaluator, parent_node, duration: TotalDurationInfo, sams: SamsClubHours) -> None:
    q_node = evaluator.add_parallel(
        id="Question_4_Total_Sams_Club_Shopping_Hours_Plus",
        desc="Calculates the total number of hours available to shop at Sam's Club on Black Friday with Plus early access, with supporting URL(s) for the underlying hours used.",
        parent=parent_node,
        critical=False,
    )

    # Verify total duration value equals 12 hours (simple verify against the answer)
    total_leaf = evaluator.add_leaf(
        id="Total_Duration_Value",
        desc="Correctly calculates the total available shopping time for Plus members from first entry (8:00 AM) to close (8:00 PM) as 12 hours.",
        parent=q_node,
        critical=True,
    )
    val = (duration.hours_value or "").strip()
    total_claim = (
        f"The total available shopping time for Sam's Club Plus members on Black Friday (Nov 28, 2025), "
        f"from first entry to close, as stated in the answer, is '{val}', and this should equal 12 hours."
    )
    await evaluator.verify(
        claim=total_claim,
        node=total_leaf,
        sources=None,  # Simple verification against the answer content
        additional_instruction=(
            "Check the stated total hours in the answer. From 8:00 AM to 8:00 PM equals 12 hours. "
            "Accept reasonable forms like '12', '12 hours', or 'twelve hours'."
        )
    )

    # Reference URLs existence (can reuse Sam's Club hour sources)
    refs_leaf = evaluator.add_leaf(
        id="Duration_Reference_URLs",
        desc="Provides supporting reference URL(s) (can reuse the Sam's Club hours sources) for the hours used in the duration calculation.",
        parent=q_node,
        critical=True,
    )
    refs_claim = (
        "The provided sources substantiate the Sam's Club Black Friday hours used for the duration calculation "
        "(Plus start 8:00 AM and store close 8:00 PM)."
    )
    # Prefer 'duration.sources' if provided; if empty, reuse sams.sources
    duration_sources = duration.sources if duration.sources else sams.sources
    await evaluator.verify(
        claim=refs_claim,
        node=refs_leaf,
        sources=duration_sources,
        additional_instruction="Verify that the URLs support the hours used (Plus 8:00 AM, close 8:00 PM) for computing total shopping duration."
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
    Evaluate the answer for the Thanksgiving weekend shopping plan task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root: parallel across the four questions
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

    # Extract all parts
    wegmans_info = await evaluator.extract(
        prompt=prompt_extract_wegmans(),
        template_class=WegmansInfo,
        extraction_name="wegmans_info",
    )
    chipotle_info = await evaluator.extract(
        prompt=prompt_extract_chipotle(),
        template_class=ChipotlePromo,
        extraction_name="chipotle_promo",
    )
    sams_info = await evaluator.extract(
        prompt=prompt_extract_sams(),
        template_class=SamsClubHours,
        extraction_name="sams_club_hours",
    )
    duration_info = await evaluator.extract(
        prompt=prompt_extract_total_duration(),
        template_class=TotalDurationInfo,
        extraction_name="total_duration_info",
    )

    # Optional ground truth/context info
    evaluator.add_ground_truth({
        "expected_values": {
            "chipotle_promotion_name": EXPECTED_CHIPOTLE_PROMO_NAME,
            "sams_plus_member_start": EXPECTED_SAMS_PLUS_START,
            "sams_regular_member_start": EXPECTED_SAMS_REGULAR_START,
            "sams_store_closing_time": EXPECTED_SAMS_CLOSE,
            "expected_total_hours": EXPECTED_TOTAL_HOURS
        },
        "dates": {
            "thanksgiving_eve": "November 26, 2025",
            "thanksgiving_day": "November 27, 2025",
            "black_friday": "November 28, 2025"
        }
    })

    # Build verification tree for each question
    await build_wegmans_verification(evaluator, root, wegmans_info)
    await build_chipotle_verification(evaluator, root, chipotle_info)
    await build_sams_verification(evaluator, root, sams_info)
    await build_duration_verification(evaluator, root, duration_info, sams_info)

    # Return summary
    return evaluator.get_summary()