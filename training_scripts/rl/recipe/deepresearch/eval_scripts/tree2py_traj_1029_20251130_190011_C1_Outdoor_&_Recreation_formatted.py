import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sdz_dec2025_trip_planning"
TASK_DESCRIPTION = (
    "You are planning to visit the San Diego Zoo in December 2025. Provide the following information to help with trip planning: "
    "(1) What is the adult (ages 12 and older) admission price for a 1-Day Pass (Any Day ticket)? "
    "(2) What is the parking cost at the Zoo during December 2025? "
    "(3) Is the San Diego Zoo open on major holidays like Christmas Day (December 25, 2025)?"
)

GROUND_TRUTH = {
    "adult_any_day_price_usd": 76,
    "parking_2025_general_is_free": True,
    "parking_fees_begin": "January 5, 2026",
    "december_nights_2025_dates": "Dec 5–6, 2025",
    "december_nights_parking_fee_usd": 35,
    "december_nights_waiver_for_ticket_holders_and_members": True,
    "open_every_day_including_christmas": True,
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PriceInfo(BaseModel):
    amount_str: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ParkingGeneralRule(BaseModel):
    mentions_free_2025: Optional[bool] = None
    mentions_fee_start_jan5_2026: Optional[bool] = None
    statement_str: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DecemberNightsInfo(BaseModel):
    mentions_exception: Optional[bool] = None
    dates_str: Optional[str] = None
    price_str: Optional[str] = None
    waived_for_ticket_holders_members: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class HolidayStatusInfo(BaseModel):
    mentions_open_every_day_including_christmas: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class ZooPlanningExtraction(BaseModel):
    adult_price: Optional[PriceInfo] = None
    parking_general: Optional[ParkingGeneralRule] = None
    december_nights: Optional[DecemberNightsInfo] = None
    holiday_status: Optional[HolidayStatusInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_zoo_planning() -> str:
    return (
        "Extract structured information from the answer for planning a December 2025 visit to the San Diego Zoo.\n"
        "Return a JSON object with these nested fields:\n"
        "adult_price:\n"
        "  - amount_str: The adult (ages 12+) 1-Day Pass (Any Day ticket) price as stated in the answer (e.g., '$76' or '76'). If missing, return null.\n"
        "  - sources: All URLs in the answer that specifically support or relate to the adult ticket price claim.\n"
        "parking_general:\n"
        "  - mentions_free_2025: true/false depending on whether the answer states Zoo parking is free in 2025 (including December 2025). If not stated, return null.\n"
        "  - mentions_fee_start_jan5_2026: true/false depending on whether the answer states that paid parking/fees begin January 5, 2026. If not stated, return null.\n"
        "  - statement_str: The exact sentence or short summary provided in the answer about general 2025 parking and fee start.\n"
        "  - sources: All URLs in the answer that support the general parking rule and fee start statement.\n"
        "december_nights:\n"
        "  - mentions_exception: true/false depending on whether the answer mentions an exception for 'December Nights'. If not mentioned, return null.\n"
        "  - dates_str: The dates stated for December Nights 2025 (e.g., 'Dec 5–6, 2025'). If missing, return null.\n"
        "  - price_str: The parking price per vehicle stated for December Nights (e.g., '$35'). If missing, return null.\n"
        "  - waived_for_ticket_holders_members: true/false depending on whether the answer says parking is waived for ticket holders and members during December Nights.\n"
        "  - sources: All URLs in the answer that support the December Nights parking exception details.\n"
        "holiday_status:\n"
        "  - mentions_open_every_day_including_christmas: true/false depending on whether the answer states the Zoo is open every day including Christmas Day (Dec 25, 2025). If not stated, return null.\n"
        "  - sources: All URLs in the answer that support the holiday operating status.\n"
        "If any URL is mentioned in markdown link format, extract the actual URL. If no URL is given for a section, return an empty list for sources."
    )


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root: Any,
    extracted: ZooPlanningExtraction,
) -> None:
    # Create the critical parent node representing the overall visit information
    visit_info_node = evaluator.add_parallel(
        id="San_Diego_Zoo_Visit_Information",
        desc="Verification of complete and accurate information for planning a December 2025 visit to the San Diego Zoo",
        parent=root,
        critical=True,
    )

    # 1) Adult Admission Price: must state $76 for 1-Day Pass (Any Day) adults (12+)
    adult_price_leaf = evaluator.add_leaf(
        id="Adult_Admission_Price",
        desc="States the adult (ages 12+) San Diego Zoo 1-Day Pass (Any Day ticket) admission price as $76",
        parent=visit_info_node,
        critical=True,
    )

    # Verify that the answer states $76; focus on the answer content (no URL required for this check)
    await evaluator.verify(
        claim="The answer states that the adult (ages 12+) 1-Day Pass (Any Day ticket) admission price is $76.",
        node=adult_price_leaf,
        sources=None,
        additional_instruction=(
            "Check the answer text itself for the dollar amount specifically tied to the adult 1-Day Pass (Any Day ticket). "
            "Consider minor formatting variations ($76 vs 76), but the numeric value must be 76. "
            "If the answer lists a different price or omits the adult Any Day ticket price, mark this incorrect."
        ),
    )

    # 2) Parking Cost in December 2025: general rule + December Nights exception
    parking_node = evaluator.add_parallel(
        id="December_2025_Parking_Cost",
        desc="Correctly describes San Diego Zoo parking costs during December 2025 per constraints (general rule + any stated exception(s))",
        parent=visit_info_node,
        critical=True,
    )

    # 2a) General Parking Rule in 2025 (and fees begin Jan 5, 2026)
    general_parking_leaf = evaluator.add_leaf(
        id="General_Parking_Rule_2025",
        desc="States that San Diego Zoo parking is free in 2025 (thus during December 2025) and that parking fees begin January 5, 2026",
        parent=parking_node,
        critical=True,
    )
    general_sources = extracted.parking_general.sources if (extracted.parking_general and extracted.parking_general.sources) else []
    await evaluator.verify(
        claim=(
            "San Diego Zoo parking is free in 2025, including December 2025, and paid parking/fees begin on January 5, 2026."
        ),
        node=general_parking_leaf,
        sources=general_sources,
        additional_instruction=(
            "Use the provided URLs (e.g., official San Diego Zoo pages) to confirm the 2025 free-parking policy and the specific start date for paid parking (January 5, 2026). "
            "If any URL is irrelevant or does not explicitly state these details, mark as not supported."
        ),
    )

    # 2b) December Nights Exception (Dec 5–6, 2025, $35 per vehicle, waived for ticket holders and members)
    december_nights_leaf = evaluator.add_leaf(
        id="December_Nights_Exception",
        desc="Mentions that during December Nights (Dec 5–6, 2025) parking is $35 per vehicle, but waived for ticket holders and members",
        parent=parking_node,
        critical=True,
    )
    december_sources = extracted.december_nights.sources if (extracted.december_nights and extracted.december_nights.sources) else []
    await evaluator.verify(
        claim=(
            "During December Nights (Dec 5–6, 2025), parking is $35 per vehicle, and this fee is waived for San Diego Zoo ticket holders and members."
        ),
        node=december_nights_leaf,
        sources=december_sources,
        additional_instruction=(
            "Verify the event dates, the $35 parking fee per vehicle, and the waiver for ticket holders and members, using the provided URLs "
            "(e.g., Zoo or Balboa Park event/parking pages). If any of these elements are not present or contradicted, mark as not supported."
        ),
    )

    # 3) Holiday Operating Status: open every day including Christmas Day (Dec 25, 2025)
    holiday_leaf = evaluator.add_leaf(
        id="Holiday_Operating_Status",
        desc="States the San Diego Zoo is open every day of the year, including holidays such as Christmas Day (Dec 25, 2025)",
        parent=visit_info_node,
        critical=True,
    )
    holiday_sources = extracted.holiday_status.sources if (extracted.holiday_status and extracted.holiday_status.sources) else []
    await evaluator.verify(
        claim=(
            "The San Diego Zoo is open every day of the year, including holidays such as Christmas Day (December 25, 2025)."
        ),
        node=holiday_leaf,
        sources=holiday_sources,
        additional_instruction=(
            "Confirm via the provided URLs (e.g., Zoo hours/visit pages) that the Zoo is open daily, including Christmas Day."
        ),
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
    model: str = "o4-mini",
) -> Dict:
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_zoo_planning(),
        template_class=ZooPlanningExtraction,
        extraction_name="zoo_planning_extraction",
    )

    evaluator.add_ground_truth(
        {
            "adult_any_day_price_usd": GROUND_TRUTH["adult_any_day_price_usd"],
            "parking_general_is_free_2025": GROUND_TRUTH["parking_2025_general_is_free"],
            "parking_fees_begin": GROUND_TRUTH["parking_fees_begin"],
            "december_nights_2025_dates": GROUND_TRUTH["december_nights_2025_dates"],
            "december_nights_parking_fee_usd": GROUND_TRUTH["december_nights_parking_fee_usd"],
            "december_nights_waiver_for_ticket_holders_and_members": GROUND_TRUTH[
                "december_nights_waiver_for_ticket_holders_and_members"
            ],
            "open_every_day_including_christmas": GROUND_TRUTH["open_every_day_including_christmas"],
        },
        gt_type="expected_values",
    )

    await build_verification_tree(evaluator, root, extracted)

    return evaluator.get_summary()