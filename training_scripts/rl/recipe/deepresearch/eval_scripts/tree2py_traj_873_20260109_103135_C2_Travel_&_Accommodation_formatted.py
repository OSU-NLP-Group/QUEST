import asyncio
import logging
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "us_germany_passport_hotel"
TASK_DESCRIPTION = "A US citizen is planning a 30-day trip to Germany in June 2026. What is the passport validity requirement for entering Germany, and what are two standard requirements they should expect when checking into a hotel?"


class PassportValidity(BaseModel):
    text: Optional[str] = None
    mentions_3_months_beyond_departure: Optional[bool] = None


class HotelIDRequirement(BaseModel):
    text: Optional[str] = None
    mentions_name_match: Optional[bool] = None


class CreditCardHold(BaseModel):
    core_text: Optional[str] = None
    mentions_amount_typical: Optional[bool] = None
    amount_text: Optional[str] = None
    mentions_release_timeframe: Optional[bool] = None
    release_timeframe_text: Optional[str] = None


class TravelRequirementsExtraction(BaseModel):
    passport_validity: Optional[PassportValidity] = None
    hotel_id: Optional[HotelIDRequirement] = None
    credit_hold: Optional[CreditCardHold] = None


def prompt_extract_requirements() -> str:
    return """
    Extract the specific statements the answer makes regarding:
    1) Passport validity for entering Germany/Schengen.
       - text: the exact sentence or phrase about passport validity.
       - mentions_3_months_beyond_departure: true if the answer explicitly states that the passport must be valid for at least 3 months beyond the intended departure date from the Schengen Area (accept equivalent phrasing such as "90 days after leaving Schengen").
    2) Hotel check-in ID requirements.
       - text: the sentence stating hotels require a government-issued photo ID (passport/driver's license).
       - mentions_name_match: true if the answer states the ID name must match the reservation/booking name (accept equivalent phrasing).
    3) Hotel credit card pre-authorization/hold practice.
       - core_text: the sentence(s) saying hotels place a temporary hold/pre-authorization at check-in to cover incidentals/potential damages.
       - mentions_amount_typical: true if the answer mentions typical hold sizing (e.g., one night's stay, fixed amount, or a percentage).
       - amount_text: the exact phrase for the typical hold sizing if present.
       - mentions_release_timeframe: true if the answer mentions typical release timing (about 3–10 business days after check-out, depending on bank/provider; accept equivalent phrasing like "a few business days" or "up to a week").
       - release_timeframe_text: the exact phrase for the release timing if present.

    Important:
    - Only extract what is explicitly stated in the answer text.
    - If a specific detail is not present, set the corresponding field to null or false.
    """


async def build_passport_validity_subtree(evaluator: Evaluator, parent_node) -> None:
    passport_node = evaluator.add_parallel(
        id="Passport_Validity_Requirement",
        desc="States the passport validity requirement for entering Germany/Schengen as specified in constraints.",
        parent=parent_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Passport_Validity_3_Months_Beyond_Departure",
        desc="States passport must be valid for at least 3 months beyond the intended departure date from the Schengen Area.",
        parent=passport_node,
        critical=True
    )

    claim = (
        "The answer explicitly states that the passport must be valid for at least 3 months beyond "
        "the intended departure date from the Schengen Area (accept equivalent phrasing like '90 days after leaving Schengen')."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction=(
            "Focus only on whether the answer mentions this specific requirement. "
            "Accept reasonable synonyms or paraphrases that clearly mean 3 months beyond departure. "
            "If the answer uses a '6-month validity' rule without also stating the 3-month Schengen rule, treat it as not meeting this requirement."
        ),
    )


async def build_hotel_requirements_subtree(evaluator: Evaluator, parent_node) -> None:
    hotel_node = evaluator.add_parallel(
        id="Hotel_CheckIn_Requirements",
        desc="Provides two standard hotel check-in requirements per constraints (ID verification and credit card pre-authorization/hold).",
        parent=parent_node,
        critical=True
    )

    id_leaf = evaluator.add_leaf(
        id="ID_Verification",
        desc="States hotels require government-issued photo ID at check-in and that the ID name matches the reservation.",
        parent=hotel_node,
        critical=True
    )
    id_claim = (
        "The answer states that hotels require a government-issued photo ID (e.g., passport or driver's license) at check-in "
        "and also states that the name on the ID must match the reservation/booking."
    )
    await evaluator.verify(
        claim=id_claim,
        node=id_leaf,
        additional_instruction=(
            "Check the answer for both parts: (1) photo ID required and (2) name on ID must match the reservation. "
            "Accept equivalent phrasing like 'ID must match the booking name'."
        ),
    )

    core_hold_leaf = evaluator.add_leaf(
        id="Hold_Core_Requirement",
        desc="States hotels typically place a credit card hold/pre-authorization at check-in to cover incidentals/potential damages.",
        parent=hotel_node,
        critical=True
    )
    core_hold_claim = (
        "The answer states that hotels typically place a credit card pre-authorization or temporary hold at check-in "
        "to cover incidentals and potential damages."
    )
    await evaluator.verify(
        claim=core_hold_claim,
        node=core_hold_leaf,
        additional_instruction=(
            "Accept synonyms such as 'deposit', 'authorization', 'temporary hold', 'security hold'. "
            "It must clearly be at check-in and for incidentals/damages."
        ),
    )


async def build_credit_hold_optional_details(evaluator: Evaluator, parent_node) -> None:
    # Optional, non-critical details about the credit card hold
    details_node = evaluator.add_parallel(
        id="Credit_Card_Hold",
        desc="Evaluates whether the answer describes the standard credit card pre-authorization/hold practice at hotel check-in.",
        parent=parent_node,
        critical=False
    )

    amount_leaf = evaluator.add_leaf(
        id="Hold_Amount_Typical",
        desc="Mentions typical hold sizing (e.g., one night's stay or a fixed percentage) consistent with constraints.",
        parent=details_node,
        critical=False
    )
    amount_claim = (
        "The answer mentions a typical hold amount sizing, such as one night's stay, a fixed dollar amount, or a percentage of the booking."
    )
    await evaluator.verify(
        claim=amount_claim,
        node=amount_leaf,
        additional_instruction=(
            "Accept examples like 'one night’s room rate', '$50-$200', or 'a percentage of the total'. "
            "If no typical sizing is mentioned, this should be marked incorrect."
        ),
    )

    release_leaf = evaluator.add_leaf(
        id="Hold_Release_Timeframe",
        desc="Mentions typical release timing (about 3–10 business days after check-out, depending on bank/provider) consistent with constraints.",
        parent=details_node,
        critical=False
    )
    release_claim = (
        "The answer mentions a typical release timeframe for the credit card hold, approximately 3 to 10 business days after check-out, "
        "depending on the bank or card provider."
    )
    await evaluator.verify(
        claim=release_claim,
        node=release_leaf,
        additional_instruction=(
            "Accept equivalent phrasing such as 'a few business days', 'up to a week', or 'depends on your bank'. "
            "If no timing is mentioned, this should be marked incorrect."
        ),
    )


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
        prompt=prompt_extract_requirements(),
        template_class=TravelRequirementsExtraction,
        extraction_name="requirements_extraction"
    )

    evaluator.add_ground_truth({
        "constraints": {
            "passport_validity": "At least 3 months beyond intended departure from the Schengen Area.",
            "hotel_check_in": [
                "Government-issued photo ID required; ID name should match reservation.",
                "Credit card pre-authorization/temporary hold at check-in for incidentals/damages."
            ],
            "optional_details": {
                "hold_amount_typical": "Often one night's stay, fixed amount, or a percentage.",
                "hold_release_timeframe": "Typically about 3–10 business days after check-out depending on bank/provider."
            }
        },
        "trip_context": {
            "citizenship": "US",
            "destination": "Germany (Schengen)",
            "duration": "30 days",
            "month_year": "June 2026"
        }
    })

    await build_passport_validity_subtree(evaluator, root)
    await build_hotel_requirements_subtree(evaluator, root)
    await build_credit_hold_optional_details(evaluator, root)

    return evaluator.get_summary()