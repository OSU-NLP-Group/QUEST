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
TASK_ID = "marathon_world_records"
TASK_DESCRIPTION = """
Who currently holds the men's marathon world record, and what are the details of this record including the time, location, and date it was set? Additionally, who holds the T12 (visually impaired) men's marathon world record, and what are the details of this record including the time, location, and date?
"""

# Ground truth (for summary/reference only; verification uses cited sources)
GROUND_TRUTH = {
    "mens_record": {
        "holder_name": "Kelvin Kiptum",
        "nationality": "Kenya",
        "time": "2:00:35",
        "location": "Chicago Marathon",
        "date": "October 8, 2023"
    },
    "t12_record": {
        "holder_name": "Jaryd Clifford",
        "nationality": "Australia",
        "time": "2:19:08",
        "location": "Sydney",
        "date": "April 25, 2021"
    },
    "t12_classification": {
        "is_visual_impairment": True,
        "eligibility_criteria_text": "Visual field of less than 5 degrees radius and/or visual acuity in LogMAR 1.50–2.60 range"
    }
}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class MarathonRecord(BaseModel):
    holder_name: Optional[str] = None
    nationality: Optional[str] = None
    time: Optional[str] = None
    location: Optional[str] = None
    date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class T12Classification(BaseModel):
    classification_statement: Optional[str] = None
    eligibility_criteria: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RecordsExtraction(BaseModel):
    mens: Optional[MarathonRecord] = None
    t12_record: Optional[MarathonRecord] = None
    t12_class: Optional[T12Classification] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_world_records() -> str:
    return """
    Extract the structured details the answer provides for:
    1) The current men's marathon world record, and
    2) The T12 (visually impaired) men's marathon world record, as well as the T12 classification details.

    For each record (men's and T12), extract:
    - holder_name: The person's name stated as the record holder.
    - nationality: The nationality of the record holder (as stated).
    - time: The record time (e.g., '2:00:35').
    - location: The event/location where the record was set (e.g., 'Chicago Marathon' or 'Sydney').
    - date: The date when the record was set (e.g., 'October 8, 2023' or 'April 25, 2021').
    - sources: A list of URLs explicitly cited in the answer that support this record and its details.

    For the T12 classification, extract:
    - classification_statement: The sentence or statement from the answer that describes the T12 classification (e.g., that T12 is for athletes with visual impairment).
    - eligibility_criteria: The textual description of eligibility thresholds mentioned (e.g., 'visual field of less than 5 degrees radius and/or visual acuity in the LogMAR 1.50–2.60 range').
    - sources: A list of URLs explicitly cited in the answer for the T12 classification/criteria.

    Return a JSON object with the following top-level fields:
    {
      "mens": { ... },          // MarathonRecord
      "t12_record": { ... },    // MarathonRecord
      "t12_class": { ... }      // T12Classification
    }

    RULES:
    - Only extract information explicitly present in the answer.
    - If any field is missing, set it to null (for strings) or an empty list (for sources).
    - Extract actual URLs from the answer; include full URLs with protocol.
    - Do not invent information or URLs that are not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_mens_record(
    evaluator: Evaluator,
    parent_node,
    mens: Optional[MarathonRecord],
) -> None:
    """
    Build verification nodes for the current men's marathon world record subtree.
    All leaves are critical and verified against the cited sources from the answer.
    """
    mens_node = evaluator.add_parallel(
        id="Mens_Marathon_World_Record",
        desc="Information about the current men's marathon world record holder",
        parent=parent_node,
        critical=True
    )

    # Existence of sources gating (critical)
    mens_sources_exist = evaluator.add_custom_node(
        result=bool(mens and mens.sources and len(mens.sources) > 0),
        id="Mens_Sources_Provided",
        desc="Men's marathon world record: sources are provided in the answer",
        parent=mens_node,
        critical=True
    )

    sources_list = mens.sources if mens else []

    # Record holder (name + nationality)
    holder_node = evaluator.add_leaf(
        id="Mens_Record_Holder",
        desc="Correctly identify the current men's marathon world record holder's name and nationality",
        parent=mens_node,
        critical=True
    )
    holder_name = mens.holder_name if mens and mens.holder_name else ""
    nationality = mens.nationality if mens and mens.nationality else ""
    holder_claim = f"{holder_name} from {nationality} holds the current men's marathon world record."
    await evaluator.verify(
        claim=holder_claim,
        node=holder_node,
        sources=sources_list,
        additional_instruction="Verify that the cited page(s) explicitly indicate the current men's marathon world record holder and their nationality. Allow reasonable variants like 'Kenyan' vs 'Kenya'.",
        extra_prerequisites=[mens_sources_exist]
    )

    # Record time
    time_node = evaluator.add_leaf(
        id="Mens_Record_Time",
        desc="Provide the correct world record time",
        parent=mens_node,
        critical=True
    )
    time_str = mens.time if mens and mens.time else ""
    time_claim = f"The current men's marathon world record time is {time_str}."
    await evaluator.verify(
        claim=time_claim,
        node=time_node,
        sources=sources_list,
        additional_instruction="Check the authoritative record page(s) for the marathon world record time. Accept minor formatting variations (e.g., leading zeros).",
        extra_prerequisites=[mens_sources_exist]
    )

    # Location / event
    location_node = evaluator.add_leaf(
        id="Mens_Location",
        desc="Provide the correct location/event where the record was set",
        parent=mens_node,
        critical=True
    )
    location_str = mens.location if mens and mens.location else ""
    location_claim = f"The world record was set at the {location_str}."
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=sources_list,
        additional_instruction="Verify that the page(s) indicate the event/location where the record occurred (e.g., Chicago Marathon). Allow minor naming variants.",
        extra_prerequisites=[mens_sources_exist]
    )

    # Date
    date_node = evaluator.add_leaf(
        id="Mens_Date",
        desc="Provide the correct date when the record was set",
        parent=mens_node,
        critical=True
    )
    date_str = mens.date if mens and mens.date else ""
    date_claim = f"The world record was set on {date_str}."
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=sources_list,
        additional_instruction="Verify the date on the cited page(s); allow reasonable date format variants (e.g., '8 October 2023' vs 'October 8, 2023').",
        extra_prerequisites=[mens_sources_exist]
    )


async def verify_t12_section(
    evaluator: Evaluator,
    parent_node,
    t12_class: Optional[T12Classification],
    t12_record: Optional[MarathonRecord],
) -> None:
    """
    Build verification nodes for the T12 (visually impaired) men's marathon world record subtree.
    Includes both classification details and record details. All leaves are critical.
    """
    t12_node = evaluator.add_parallel(
        id="T12_Marathon_World_Record",
        desc="Information about the T12 (visually impaired) men's marathon world record holder",
        parent=parent_node,
        critical=True
    )

    # --- Classification part ---
    class_sources_exist = evaluator.add_custom_node(
        result=bool(t12_class and t12_class.sources and len(t12_class.sources) > 0),
        id="T12_Classification_Sources_Provided",
        desc="T12 classification: sources are provided in the answer",
        parent=t12_node,
        critical=True
    )
    class_sources = t12_class.sources if t12_class else []

    # T12 classification is for visual impairment
    t12_vis_imp_node = evaluator.add_leaf(
        id="T12_Classification_Visual_Impairment",
        desc="State that the T12 Paralympic classification is for athletes with visual impairment",
        parent=t12_node,
        critical=True
    )
    vis_imp_claim = "The T12 Paralympic classification is for athletes with visual impairment."
    await evaluator.verify(
        claim=vis_imp_claim,
        node=t12_vis_imp_node,
        sources=class_sources,
        additional_instruction="Verify via classification documentation (e.g., World Para Athletics) that T12 covers athletes with visual impairment.",
        extra_prerequisites=[class_sources_exist]
    )

    # T12 eligibility criteria thresholds
    t12_elig_node = evaluator.add_leaf(
        id="T12_Eligibility_Criteria",
        desc="State the T12 eligibility criteria thresholds: visual field of less than 5 degrees radius and/or visual acuity in the LogMAR 1.50–2.60 range",
        parent=t12_node,
        critical=True
    )
    elig_claim = "T12 eligibility thresholds include a visual field of less than 5 degrees radius and/or visual acuity in the LogMAR 1.50–2.60 range."
    await evaluator.verify(
        claim=elig_claim,
        node=t12_elig_node,
        sources=class_sources,
        additional_instruction="Verify that the classification documents state the thresholds (visual field < 5 degrees radius and/or visual acuity LogMAR 1.50–2.60). Allow wording variants and hyphen/en dash variations.",
        extra_prerequisites=[class_sources_exist]
    )

    # --- T12 record part ---
    record_sources_exist = evaluator.add_custom_node(
        result=bool(t12_record and t12_record.sources and len(t12_record.sources) > 0),
        id="T12_Record_Sources_Provided",
        desc="T12 men's marathon record: sources are provided in the answer",
        parent=t12_node,
        critical=True
    )
    record_sources = t12_record.sources if t12_record else []

    # Record holder (name + nationality)
    t12_holder_node = evaluator.add_leaf(
        id="T12_Record_Holder",
        desc="Correctly identify the T12 men's marathon world record holder's name and nationality",
        parent=t12_node,
        critical=True
    )
    t12_holder_name = t12_record.holder_name if t12_record and t12_record.holder_name else ""
    t12_nationality = t12_record.nationality if t12_record and t12_record.nationality else ""
    t12_holder_claim = f"{t12_holder_name} from {t12_nationality} holds the T12 men's marathon world record."
    await evaluator.verify(
        claim=t12_holder_claim,
        node=t12_holder_node,
        sources=record_sources,
        additional_instruction="Verify that the cited page(s) explicitly state the T12 men's marathon record holder and their nationality. Allow reasonable variants.",
        extra_prerequisites=[record_sources_exist]
    )

    # Record time
    t12_time_node = evaluator.add_leaf(
        id="T12_Record_Time",
        desc="Provide the correct T12 world record time",
        parent=t12_node,
        critical=True
    )
    t12_time_str = t12_record.time if t12_record and t12_record.time else ""
    t12_time_claim = f"The T12 men's marathon world record time is {t12_time_str}."
    await evaluator.verify(
        claim=t12_time_claim,
        node=t12_time_node,
        sources=record_sources,
        additional_instruction="Verify the T12 marathon record time from authoritative or event sources; allow minor formatting variations.",
        extra_prerequisites=[record_sources_exist]
    )

    # Location
    t12_location_node = evaluator.add_leaf(
        id="T12_Location",
        desc="Provide the correct location where the T12 record was set",
        parent=t12_node,
        critical=True
    )
    t12_location_str = t12_record.location if t12_record and t12_record.location else ""
    t12_location_claim = f"The T12 men's marathon world record was set in {t12_location_str}."
    await evaluator.verify(
        claim=t12_location_claim,
        node=t12_location_node,
        sources=record_sources,
        additional_instruction="Verify the location/event stated for the T12 marathon record.",
        extra_prerequisites=[record_sources_exist]
    )

    # Date
    t12_date_node = evaluator.add_leaf(
        id="T12_Date",
        desc="Provide the correct date when the T12 record was set",
        parent=t12_node,
        critical=True
    )
    t12_date_str = t12_record.date if t12_record and t12_record.date else ""
    t12_date_claim = f"The T12 men's marathon world record was set on {t12_date_str}."
    await evaluator.verify(
        claim=t12_date_claim,
        node=t12_date_node,
        sources=record_sources,
        additional_instruction="Verify the date from the cited sources; allow format variants.",
        extra_prerequisites=[record_sources_exist]
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
    Evaluate an answer for the marathon world records task.
    """
    # Initialize evaluator (root is non-critical by framework design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level aggregation parallel
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

    # Create a critical child under root to represent the overall rubric root
    overall_node = evaluator.add_parallel(
        id="Marathon_World_Records",
        desc="Provide information about the current men's marathon world record and the T12 men's marathon world record",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_world_records(),
        template_class=RecordsExtraction,
        extraction_name="world_records_extraction"
    )

    # Add ground truth information (for transparency in summary; not used to judge claims)
    evaluator.add_ground_truth({
        "mens_record_ground_truth": GROUND_TRUTH["mens_record"],
        "t12_record_ground_truth": GROUND_TRUTH["t12_record"],
        "t12_classification_ground_truth": GROUND_TRUTH["t12_classification"]
    })

    # Build and verify subtrees
    await verify_mens_record(evaluator, overall_node, extracted.mens)
    await verify_t12_section(evaluator, overall_node, extracted.t12_class, extracted.t12_record)

    # Return structured summary
    return evaluator.get_summary()