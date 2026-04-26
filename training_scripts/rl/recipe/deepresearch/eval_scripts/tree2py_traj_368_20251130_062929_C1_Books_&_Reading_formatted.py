import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "la_times_festival_of_books_2025"
TASK_DESCRIPTION = (
    "I'm interested in attending the LA Times Festival of Books in 2025. "
    "Please provide the following information: (1) the dates when the festival will be held, "
    "(2) the location or venue where it takes place, and (3) whether admission to the festival is "
    "free or requires a ticket purchase. Include a reference URL for verification."
)

EXPECTED_DATES_TEXT = "April 26–27, 2025"
EXPECTED_LOCATION_HINTS = [
    "USC", "University of Southern California", "USC campus", "USC University Park Campus", "Los Angeles"
]
EXPECTED_EVENT_NAMES = ["LA Times Festival of Books", "Los Angeles Times Festival of Books", "Festival of Books"]
EXPECTED_YEAR = "2025"


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class FestivalExtraction(BaseModel):
    """
    Structured info extracted from the agent's answer about LA Times Festival of Books 2025.
    """
    event_name: Optional[str] = None
    year: Optional[str] = None
    dates_text: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location: Optional[str] = None
    admission_text: Optional[str] = None
    admission_general_free: Optional[bool] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_festival_info() -> str:
    return """
    Extract the requested information about the LA Times Festival of Books from the answer text.

    Fields to extract:
    - event_name: The event name as stated in the answer (e.g., "LA Times Festival of Books").
    - year: The 4-digit year associated with the event in the answer (e.g., "2025").
    - dates_text: The dates string for the 2025 festival exactly as presented in the answer (e.g., "April 26–27, 2025").
    - start_date: The explicit start date if the answer provides it (e.g., "April 26, 2025"), otherwise null.
    - end_date: The explicit end date if the answer provides it (e.g., "April 27, 2025"), otherwise null.
    - location: The venue/location in the answer (e.g., "USC campus" or "University of Southern California, Los Angeles").
    - admission_text: The admission policy as described in the answer (e.g., "General admission is free. Some indoor conversations require tickets.").
    - admission_general_free: true if the answer explicitly indicates general admission is free to enter the festival grounds; false if it says admission requires buying a ticket; null if unspecified.
    - reference_urls: A list of all URLs included in the answer as references/sources for verification. Only include actual URLs explicitly present in the answer.

    Important:
    - Do not invent information. If a field is not stated, set it to null (or empty list for URLs).
    - For URL fields, include only valid http/https URLs that appear in the answer (including markdown links).
    - Keep the text values exactly as they appear in the answer (do not normalize or rephrase).
    """


# --------------------------------------------------------------------------- #
# Verification Tree Construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, parent_node, extracted: FestivalExtraction) -> None:
    """
    Build the verification tree according to the rubric and perform verifications.
    """

    # Main (critical) node corresponding to the rubric root
    main_node = evaluator.add_parallel(
        id="LA_Times_Festival_of_Books_2025_Information",
        desc="Evaluate whether the response correctly provides dates, location, admission status, and a verifying reference for the LA Times Festival of Books 2025.",
        parent=parent_node,
        critical=True,
    )

    # 1) Correct Event and Year (leaf)
    node_event_year = evaluator.add_leaf(
        id="Correct_Event_and_Year",
        desc="Response pertains to the LA Times Festival of Books and specifically the 2025 instance.",
        parent=main_node,
        critical=True,
    )
    claim_event_year = (
        "The answer explicitly pertains to the Los Angeles Times Festival of Books (aka LA Times Festival of Books) "
        "and specifically refers to the 2025 instance."
    )
    await evaluator.verify(
        claim=claim_event_year,
        node=node_event_year,
        additional_instruction=(
            "Judge based solely on the answer text. Accept minor naming variants such as "
            "'LA Times Festival of Books', 'Los Angeles Times Festival of Books', or 'Festival of Books'. "
            "Ensure that the year 2025 is clearly stated or unambiguously indicated."
        ),
    )

    # 2) Correct Festival Dates (leaf)
    node_dates = evaluator.add_leaf(
        id="Correct_Festival_Dates",
        desc="Response states the festival dates as April 26–27, 2025.",
        parent=main_node,
        critical=True,
    )
    claim_dates = "The answer states the festival dates as April 26–27, 2025."
    await evaluator.verify(
        claim=claim_dates,
        node=node_dates,
        additional_instruction=(
            "Allow minor punctuation or dash variations (e.g., hyphen vs en dash) or abbreviated month (e.g., 'Apr.'). "
            "However, the meaning must clearly be April 26 and April 27, 2025."
        ),
    )

    # 3) Correct Festival Location (leaf)
    node_location = evaluator.add_leaf(
        id="Correct_Festival_Location",
        desc="Response states the festival location/venue as the USC campus (University of Southern California) in Los Angeles.",
        parent=main_node,
        critical=True,
    )
    claim_location = (
        "The answer states the festival location as the USC campus (University of Southern California) in Los Angeles."
    )
    await evaluator.verify(
        claim=claim_location,
        node=node_location,
        additional_instruction=(
            "Accept equivalent phrasings such as 'USC University Park Campus', 'USC campus in Los Angeles', "
            "or 'University of Southern California (USC), Los Angeles'."
        ),
    )

    # 4) Correct Admission Status (leaf)
    node_admission = evaluator.add_leaf(
        id="Correct_Admission_Status",
        desc="Response states that general admission is free (i.e., does not require a ticket purchase for admission).",
        parent=main_node,
        critical=True,
    )
    claim_admission = (
        "The answer states that general admission to the festival is free (no ticket purchase required to enter the festival grounds)."
    )
    await evaluator.verify(
        claim=claim_admission,
        node=node_admission,
        additional_instruction=(
            "It is acceptable if the answer also notes that select indoor conversations or stage events may require separate tickets, "
            "as long as general admission to the festival grounds is free."
        ),
    )

    # 5) Verifying Reference URL (expanded into presence + support under a critical parent)
    # Create a critical sub-node to separate URL presence and content support
    ref_parent = evaluator.add_parallel(
        id="Verifying_Reference_URL",
        desc="Response includes at least one reference URL whose content corroborates the stated dates, location, and admission status.",
        parent=main_node,
        critical=True,
    )

    # 5.a) Presence of at least one reference URL (critical leaf via custom node)
    has_ref_urls = bool(extracted.reference_urls)
    evaluator.add_custom_node(
        result=has_ref_urls,
        id="Reference_URL_Present",
        desc="At least one reference URL is included in the answer.",
        parent=ref_parent,
        critical=True,
    )

    # 5.b) Reference URL(s) support the key facts (critical leaf)
    node_ref_support = evaluator.add_leaf(
        id="Reference_URL_Supports_Facts",
        desc="At least one cited source corroborates: (a) Apr 26–27, 2025 dates; (b) USC campus in Los Angeles; (c) general admission is free.",
        parent=ref_parent,
        critical=True,
    )
    claim_refs = (
        "This webpage indicates that the Los Angeles Times Festival of Books (LA Times Festival of Books) in 2025 "
        "takes place on April 26–27, 2025, at the University of Southern California (USC) campus in Los Angeles, "
        "and that general admission is free (festival grounds are free to enter; selective indoor conversations may require separate tickets)."
    )
    await evaluator.verify(
        claim=claim_refs,
        node=node_ref_support,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Only judge based on the provided webpage(s). If any one of the cited URLs clearly supports all three facts "
            "(dates Apr 26–27, 2025; USC/University of Southern California campus in Los Angeles; general admission free), mark as supported. "
            "Allow reasonable wording variants (e.g., hyphen vs en dash in dates, 'USC' vs 'University of Southern California'). "
            "If none of the URLs are relevant or do not corroborate these facts, mark as not supported."
        ),
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
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
    Evaluate an answer for the LA Times Festival of Books 2025 information task.
    """
    # Initialize evaluator
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

    # Extraction
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_festival_info(),
        template_class=FestivalExtraction,
        extraction_name="festival_extraction",
    )

    # Add Ground Truth context for debugging/summary
    evaluator.add_ground_truth(
        {
            "expected_event_names": EXPECTED_EVENT_NAMES,
            "expected_year": EXPECTED_YEAR,
            "expected_dates_text": EXPECTED_DATES_TEXT,
            "expected_location_examples": EXPECTED_LOCATION_HINTS,
            "expected_admission": "General admission is free; select indoor conversations may require tickets.",
        },
        gt_type="expected_facts",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted_info)

    # Return structured evaluation summary
    return evaluator.get_summary()