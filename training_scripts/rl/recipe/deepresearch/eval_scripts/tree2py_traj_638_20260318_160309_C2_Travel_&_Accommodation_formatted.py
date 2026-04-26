import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "atl_to_aruba_trip_prep"
TASK_DESCRIPTION = (
    "You are planning to fly from Atlanta, Georgia to Aruba on March 25, 2026. "
    "What are the mandatory documentation requirements you must complete before departure, "
    "and what is the recommended arrival time at Hartsfield-Jackson Atlanta International Airport "
    "for your international flight?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EDCardInfo(BaseModel):
    mandatory_statement: Optional[str] = None
    timeline_statement: Optional[str] = None
    info_items: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class PassportInfo(BaseModel):
    validity_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AirportInfo(BaseModel):
    arrival_recommendation: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TripPreparationExtraction(BaseModel):
    ed_card: Optional[EDCardInfo] = None
    passport: Optional[PassportInfo] = None
    airport: Optional[AirportInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_prep() -> str:
    return """
    Extract the parts of the answer that relate to:
    1) Aruba ED Card requirements,
    2) Passport validity requirement,
    3) Recommended airport arrival time at Hartsfield-Jackson Atlanta International Airport for an international flight.

    Return a JSON object with these fields:
    - ed_card:
        - mandatory_statement: The exact phrase/sentence from the answer that indicates the Aruba ED Card is mandatory/required for entry (or null if not present).
        - timeline_statement: The exact phrase/sentence that states the timing the ED Card must be completed (e.g., "within 7 days before travel") (or null if not present).
        - info_items: A list of short phrases that the answer states are needed to complete the ED Card (e.g., "valid passport", "personal details", "contact information", "travel information", "valid credit card"). If none mentioned, return an empty list.
        - sources: All URLs in the answer that specifically pertain to the Aruba ED Card (ED card official page, tourism authority, airline guidance, etc.). If none, return an empty list.
    - passport:
        - validity_statement: The exact phrase/sentence from the answer about passport validity length for Aruba (e.g., "valid for the entire duration of stay") (or null if not present).
        - sources: All URLs that relate to passport validity for Aruba. If none, return an empty list.
    - airport:
        - arrival_recommendation: The exact phrase/sentence recommending arrival time at Hartsfield-Jackson Atlanta International Airport for the international flight (e.g., "arrive 3 hours before departure") (or null if not present).
        - sources: All URLs that relate to ATL or airline recommendations for international check-in time. If none, return an empty list.

    IMPORTANT:
    - Extract ONLY what is explicitly present in the answer.
    - Do not infer or invent. If not present, set fields to null or empty list as instructed.
    - For URLs, include any that are reasonably associated with the respective topic. Extract actual URLs (support plain or markdown formats).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_ed_card_section(
    evaluator: Evaluator,
    parent,
    extracted: TripPreparationExtraction,
) -> None:
    """
    Build and verify the ED Card documentation subtree.
    According to rubric:
      - ED_Card_Mandatory (critical)
      - ED_Card_Timeline (critical)
      - ED_Card_Information_Requirements (critical) → decomposed into 5 critical leaves for clarity
    """
    ed_node = evaluator.add_parallel(
        id="ED_Card_Documentation",
        desc="Correctly identify the Aruba ED Card requirement, its completion timeline, and the information needed to complete it",
        parent=parent,
        critical=True,  # All children under a critical node must also be critical per framework rule
    )

    # 1) ED_Card_Mandatory
    ed_mandatory_leaf = evaluator.add_leaf(
        id="ED_Card_Mandatory",
        desc="State that an Aruba ED Card (Embarkation/Disembarkation Card) is mandatory for all travelers entering Aruba",
        parent=ed_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that an Aruba ED Card (Embarkation/Disembarkation Card) is mandatory/required for travelers entering Aruba.",
        node=ed_mandatory_leaf,
        additional_instruction=(
            "Judge based on the answer text only. Accept equivalent wording such as 'required', "
            "'must complete ED Card', 'online ED card is required for entry', etc."
        ),
    )

    # 2) ED_Card_Timeline
    ed_timeline_leaf = evaluator.add_leaf(
        id="ED_Card_Timeline",
        desc="Specify that the ED Card must be completed within 7 days before the travel date",
        parent=ed_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the Aruba ED Card must be completed within 7 days before the travel date (or within the 7 days prior to arrival/departure).",
        node=ed_timeline_leaf,
        additional_instruction=(
            "Judge based on the answer text. Accept minor phrasings like 'no earlier than 7 days prior', "
            "'within seven days before travel', 'within 7 days of the trip', '7 days prior to arrival', etc. "
            "If the answer gives a different window (e.g., 3 days, 72 hours) without mentioning 7 days, mark incorrect."
        ),
    )

    # 3) ED_Card_Information_Requirements (decomposed into 5 critical checks)
    info_parent = evaluator.add_parallel(
        id="ED_Card_Information_Requirements",
        desc="Specify what information or documents are required to complete the ED Card (valid passport, personal details, contact information, travel information, and valid credit card)",
        parent=ed_node,
        critical=True,
    )

    # 3.1 Valid passport
    info_passport_leaf = evaluator.add_leaf(
        id="ED_Card_Info_Valid_Passport",
        desc="Answer lists 'valid passport' as required to complete the Aruba ED Card",
        parent=info_parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that a valid passport is needed to complete the Aruba ED Card.",
        node=info_passport_leaf,
        additional_instruction="Judge based on the answer text. Accept synonyms like 'passport details', 'passport number' that clearly imply passport information is required.",
    )

    # 3.2 Personal details
    info_personal_leaf = evaluator.add_leaf(
        id="ED_Card_Info_Personal_Details",
        desc="Answer lists 'personal details' as required to complete the Aruba ED Card",
        parent=info_parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that personal details are needed to complete the Aruba ED Card.",
        node=info_personal_leaf,
        additional_instruction="Judge based on the answer text. Accept equivalents like 'full name, date of birth', 'personal information'.",
    )

    # 3.3 Contact information
    info_contact_leaf = evaluator.add_leaf(
        id="ED_Card_Info_Contact_Information",
        desc="Answer lists 'contact information' as required to complete the Aruba ED Card",
        parent=info_parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that contact information is needed to complete the Aruba ED Card.",
        node=info_contact_leaf,
        additional_instruction="Judge based on the answer text. Accept equivalents like 'phone number, email address'.",
    )

    # 3.4 Travel information
    info_travel_leaf = evaluator.add_leaf(
        id="ED_Card_Info_Travel_Information",
        desc="Answer lists 'travel information' as required to complete the Aruba ED Card",
        parent=info_parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that travel information (e.g., itinerary/flight/lodging details) is needed to complete the Aruba ED Card.",
        node=info_travel_leaf,
        additional_instruction="Judge based on the answer text. Accept equivalents like 'flight details', 'itinerary', 'lodging info', 'travel plans'.",
    )

    # 3.5 Valid credit card
    info_cc_leaf = evaluator.add_leaf(
        id="ED_Card_Info_Valid_Credit_Card",
        desc="Answer lists 'valid credit card' as required to complete the Aruba ED Card",
        parent=info_parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that a valid credit card is needed to complete the Aruba ED Card.",
        node=info_cc_leaf,
        additional_instruction="Judge based on the answer text. Accept equivalents like 'credit card for payment/verification'.",
    )


async def verify_passport_requirement(
    evaluator: Evaluator,
    parent,
    extracted: TripPreparationExtraction,
) -> None:
    """
    Passport validity requirement (critical).
    """
    passport_leaf = evaluator.add_leaf(
        id="Passport_Requirement",
        desc="State that the passport must be valid for the entire duration of stay in Aruba",
        parent=parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the passport must be valid for the entire duration of stay in Aruba (i.e., through the departure/return date).",
        node=passport_leaf,
        additional_instruction=(
            "Judge based on the answer text. Accept equivalents such as 'valid for the length of stay', "
            "'valid for your stay', 'valid through return'. If the answer gives a different validity rule without "
            "saying 'entire duration of stay', mark incorrect."
        ),
    )


async def verify_airport_arrival_time(
    evaluator: Evaluator,
    parent,
    extracted: TripPreparationExtraction,
) -> None:
    """
    Recommended airport arrival time at Hartsfield-Jackson Atlanta International Airport for an international flight (non-critical).
    """
    airport_leaf = evaluator.add_leaf(
        id="Airport_Arrival_Time",
        desc="Recommend arriving at Atlanta airport 3 hours before the international flight departure time",
        parent=parent,
        critical=False,
    )
    await evaluator.verify(
        claim="The answer recommends arriving at Hartsfield-Jackson Atlanta International Airport around 3 hours before the international flight's departure time.",
        node=airport_leaf,
        additional_instruction=(
            "Judge based on the answer text. Accept minor variations like 'arrive at least 3 hours early', '2–3 hours', "
            "'about three hours'. Recommendations of only 2 hours or clearly less than ~3 hours should not be considered correct."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the ATL → Aruba trip preparation requirements task.
    """
    # 1) Initialize evaluator with a parallel root
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

    # 2) Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_trip_prep(),
        template_class=TripPreparationExtraction,
        extraction_name="trip_preparation_extraction",
    )

    # 3) Build rubric tree
    #    Trip Preparation Requirements (we keep as non-critical here to allow mixed children with different criticalities)
    trip_node = evaluator.add_parallel(
        id="Trip_Preparation_Requirements",
        desc="Identify and provide all mandatory and recommended requirements for traveling from Atlanta to Aruba",
        parent=root,
        critical=False,
    )

    # 3.1 ED Card subtree (critical with decomposed requirements)
    await verify_ed_card_section(evaluator, trip_node, extracted)

    # 3.2 Passport requirement (critical)
    await verify_passport_requirement(evaluator, trip_node, extracted)

    # 3.3 Airport arrival recommendation (non-critical)
    await verify_airport_arrival_time(evaluator, trip_node, extracted)

    # 4) Return summary
    return evaluator.get_summary()