import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "aera_2026_conference_details"
TASK_DESCRIPTION = (
    "A doctoral student in education is planning to attend a major academic conference in 2026 and needs to compile comprehensive "
    "information about the American Educational Research Association (AERA) 2026 Annual Meeting. Provide the following details: "
    "(1) The full conference name, host city and state, and the specific venue; "
    "(2) The exact start and end dates of the conference (including day of week, month, day, and year); "
    "(3) The name of the speaker delivering the Presidential Address and the scheduled date and time for this address; "
    "(4) The title of the Opening Plenary session; "
    "(5) The names of the speakers delivering the AERA Distinguished Lecture and the Wallace Foundation Distinguished Lecture; "
    "(6) The approximate number of sessions offered at the conference; "
    "(7) The time zone used for all conference scheduling; "
    "(8) The deadline for presenting authors to complete their registration. "
    "All information must be verified with official AERA sources or authoritative conference documentation."
)

# Ground-truth (expected canonical values per rubric)
GROUND_TRUTH = {
    "conference_name": ["AERA 2026 Annual Meeting", "American Educational Research Association 2026 Annual Meeting"],
    "city": ["Los Angeles"],
    "state": ["California", "CA"],
    "venue": ["Los Angeles Convention Center", "LACC"],
    "start_date": ["Wednesday, April 8, 2026", "Wed, April 8, 2026"],
    "end_date": ["Sunday, April 12, 2026", "Sun, April 12, 2026"],
    "presidential_address_speaker": ["Maisha T. Winn"],
    "presidential_address_date_time": [
        "Friday, April 10, 2026, 5:45 pm to 7:15 pm Pacific Time",
        "Friday, April 10, 2026, 5:45 pm–7:15 pm PT",
        "Fri, April 10, 2026, 5:45 pm–7:15 pm PT",
    ],
    "opening_plenary_title": ["Holding Fast to Histories, Holding Fast to Dreams"],
    "distinguished_lecture_speaker": ["Bryan Brayboy", "S. Bryan Brayboy"],
    "wallace_lecture_speaker": ["Bianca Baldridge", "Bianca J. Baldridge"],
    "session_count_range": ["more than 2,500 sessions", "over 2,500 sessions", "2500+ sessions", "more than 2500 sessions"],
    "time_zone": ["Pacific Time", "PT", "Pacific Daylight Time", "PDT"],
    "presenter_registration_deadline": ["February 13, 2026", "Feb 13, 2026"],
}


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class AERADetails(BaseModel):
    conference_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    venue: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    presidential_address_speaker: Optional[str] = None
    presidential_address_date_time: Optional[str] = None
    opening_plenary_title: Optional[str] = None
    distinguished_lecture_speaker: Optional[str] = None
    wallace_lecture_speaker: Optional[str] = None
    session_count_text: Optional[str] = None
    time_zone: Optional[str] = None
    presenter_registration_deadline: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_aera_details() -> str:
    return """
    Extract the following fields exactly as they appear in the provided answer text. If a field is missing, return null. Preserve the original formatting from the answer where applicable.

    Fields to extract:
    - conference_name: The full name of the conference (e.g., "AERA 2026 Annual Meeting" or "American Educational Research Association 2026 Annual Meeting").
    - city: The host city (e.g., "Los Angeles" or "Los Angeles, CA").
    - state: The host state (e.g., "California" or "CA").
    - venue: The main conference venue (e.g., "Los Angeles Convention Center" or "LACC").
    - start_date: The exact start date including day of week, month, day, and year if provided (e.g., "Wednesday, April 8, 2026").
    - end_date: The exact end date including day of week, month, day, and year if provided (e.g., "Sunday, April 12, 2026").
    - presidential_address_speaker: The name of the speaker delivering the Presidential Address.
    - presidential_address_date_time: The scheduled date and time for the Presidential Address (include timezone if present in the answer).
    - opening_plenary_title: The title of the Opening Plenary.
    - distinguished_lecture_speaker: The speaker for the AERA Distinguished Lecture.
    - wallace_lecture_speaker: The speaker for the Wallace Foundation Distinguished Lecture.
    - session_count_text: The approximate number of sessions phrased as in the answer (e.g., "more than 2,500 sessions", "2500+ sessions").
    - time_zone: The time zone used for conference scheduling (e.g., "Pacific Time", "PT", "PDT").
    - presenter_registration_deadline: The deadline date for presenting authors to complete registration (e.g., "February 13, 2026").
    - sources: An array of ALL URLs explicitly mentioned in the answer that are relevant to the AERA 2026 Annual Meeting or its official program/schedule. Include only valid URLs. Keep duplicates out.

    Rules:
    - Do not invent or infer anything not present in the answer text.
    - For URLs, extract actual links from the answer (plain URLs or markdown links). If the answer does not contain URLs, return an empty array.
    - Keep all values as strings (do not normalize or change letter case).
    """


# --------------------------------------------------------------------------- #
# Helper functions for building verification                                  #
# --------------------------------------------------------------------------- #
def _display_expected_list(values: Union[str, List[str]]) -> str:
    if isinstance(values, list):
        return "; ".join(values)
    return str(values)


async def _add_text_field_checks(
    evaluator: Evaluator,
    parent_node,
    field_id: str,
    field_human_desc: str,
    extracted_value: Optional[str],
    expected_values: List[str],
    sources: List[str],
    node_desc_from_rubric: str,
    match_tolerance_instruction: str,
    url_instruction: str,
) -> None:
    """
    Create a sequential critical sub-tree for a single text field:
    1) provided check
    2) match expected (simple verify)
    3) supported by sources (verify by urls)
    """
    # Field main node (critical, sequential)
    field_node = evaluator.add_sequential(
        id=field_id,
        desc=node_desc_from_rubric,
        parent=parent_node,
        critical=True,
    )

    # 1) Provided
    provided = evaluator.add_custom_node(
        result=(extracted_value is not None and str(extracted_value).strip() != ""),
        id=f"{field_id}_provided",
        desc=f"The answer provides {field_human_desc}",
        parent=field_node,
        critical=True,
    )

    # 2) Match expected
    match_leaf = evaluator.add_leaf(
        id=f"{field_id}_match_expected",
        desc=f"{field_human_desc} matches the expected canonical value(s)",
        parent=field_node,
        critical=True,
    )
    # Compose match claim
    expected_preview = _display_expected_list(expected_values)
    match_claim = (
        f"The provided value '{extracted_value}' is equivalent to one of the acceptable expected values: {expected_preview}."
    )
    await evaluator.verify(
        claim=match_claim,
        node=match_leaf,
        additional_instruction=match_tolerance_instruction,
    )

    # 3) Supported by sources
    support_leaf = evaluator.add_leaf(
        id=f"{field_id}_supported_by_sources",
        desc=f"{field_human_desc} is supported by the cited official/authoritative sources",
        parent=field_node,
        critical=True,
    )

    # Build a claim tailored per field description
    # We phrase it neutrally: the AERA 2026 Annual Meeting's {field} is '{value}'.
    support_claim = f"The AERA 2026 Annual Meeting {field_human_desc.lower()} is '{extracted_value}'."
    await evaluator.verify(
        claim=support_claim,
        node=support_leaf,
        sources=sources,
        additional_instruction=url_instruction,
    )


# Specialized helpers to craft instructions and claims per field
def tolerance_instruction_for_field(field_key: str) -> str:
    common = (
        "Allow minor formatting variations (e.g., capitalization, punctuation, hyphens/en dashes, "
        "use of abbreviations like CA for California or PT/PDT for Pacific Time). "
        "Treat semantically equivalent phrasings as matches."
    )
    if field_key in ("city",):
        return common + " Consider 'Los Angeles', 'Los Angeles, CA', or 'Los Angeles, California' as equivalent."
    if field_key in ("state",):
        return common + " Consider 'California' and 'CA' as equivalent."
    if field_key in ("venue",):
        return common + " Consider 'Los Angeles Convention Center' and 'LACC' as equivalent."
    if field_key in ("time_zone",):
        return common + " Consider 'Pacific Time', 'PT', 'Pacific Daylight Time', and 'PDT' as equivalent."
    if field_key in ("session_count_range", "session_count_text"):
        return common + " Consider 'more than 2,500 sessions', 'over 2,500 sessions', '2500+ sessions' as equivalent."
    if field_key in ("presidential_address_date_time",):
        return common + " Accept minor formatting differences in time range (e.g., '5:45 pm–7:15 pm', '17:45–19:15'), date shorthands, and PT/PDT."
    return common


def url_instruction_for_field(field_key: str) -> str:
    base = (
        "Use only the webpage(s) provided. Determine whether the page(s) explicitly support the claim about the AERA 2026 Annual Meeting. "
        "Prefer official AERA pages (e.g., aera.net, aera.confex.com, official program/schedule platforms) or authoritative conference "
        "documentation. Allow minor textual variations and check screenshots if the text alone seems insufficient."
    )
    if field_key in ("city", "state"):
        return base + " If a page lists the host city/state only once, that still suffices."
    if field_key == "venue":
        return base + " Confirm that the venue is the Los Angeles Convention Center (LACC)."
    if field_key in ("start_date", "end_date"):
        return base + " Confirm exact dates; minor weekday formatting variations are acceptable."
    if field_key == "presidential_address_speaker":
        return base + " Confirm that Maisha T. Winn is delivering the Presidential Address."
    if field_key == "presidential_address_date_time":
        return base + " Confirm the date and the time window for the Presidential Address in Pacific Time."
    if field_key == "opening_plenary_title":
        return base + " Confirm the Opening Plenary title."
    if field_key == "distinguished_lecture_speaker":
        return base + " Confirm the AERA Distinguished Lecture speaker."
    if field_key == "wallace_lecture_speaker":
        return base + " Confirm the Wallace Foundation Distinguished Lecture speaker."
    if field_key == "session_count_text":
        return base + " Confirm that the meeting features more than 2,500 sessions (accept similar phrasings like '2500+')."
    if field_key == "time_zone":
        return base + " Confirm that all conference times are in Pacific Time (PT/PDT)."
    if field_key == "presenter_registration_deadline":
        return base + " Confirm the deadline date for presenting authors to complete registration."
    if field_key == "conference_name":
        return base + " Confirm an official naming equivalent to the provided string."
    return base


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: AERADetails) -> None:
    """
    Build the verification tree for AERA 2026 details based on the rubric.
    """
    # Parent group node (critical, parallel)
    aera_root = evaluator.add_parallel(
        id="AERA_2026_Conference_Details",
        desc="Verify comprehensive details about the AERA 2026 Annual Meeting",
        parent=evaluator.root,
        critical=True,
    )

    # Global gate: the answer should include at least one source URL
    sources = extracted.sources or []
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="official_sources_provided",
        desc="At least one source URL is provided in the answer to support claims (preferably official AERA or authoritative conference documentation).",
        parent=aera_root,
        critical=True,
    )

    # Conference Name
    await _add_text_field_checks(
        evaluator,
        aera_root,
        field_id="Conference_Name",
        field_human_desc="full conference name",
        extracted_value=extracted.conference_name,
        expected_values=GROUND_TRUTH["conference_name"],
        sources=sources,
        node_desc_from_rubric="Correctly identify the full conference name as 'AERA 2026 Annual Meeting' or 'American Educational Research Association 2026 Annual Meeting'",
        match_tolerance_instruction=tolerance_instruction_for_field("conference_name"),
        url_instruction=url_instruction_for_field("conference_name"),
    )

    # City
    await _add_text_field_checks(
        evaluator,
        aera_root,
        field_id="Conference_City",
        field_human_desc="host city",
        extracted_value=extracted.city,
        expected_values=GROUND_TRUTH["city"],
        sources=sources,
        node_desc_from_rubric="Correctly identify Los Angeles as the host city",
        match_tolerance_instruction=tolerance_instruction_for_field("city"),
        url_instruction=url_instruction_for_field("city"),
    )

    # State
    await _add_text_field_checks(
        evaluator,
        aera_root,
        field_id="Conference_State",
        field_human_desc="host state",
        extracted_value=extracted.state,
        expected_values=GROUND_TRUTH["state"],
        sources=sources,
        node_desc_from_rubric="Correctly identify California as the host state",
        match_tolerance_instruction=tolerance_instruction_for_field("state"),
        url_instruction=url_instruction_for_field("state"),
    )

    # Venue
    await _add_text_field_checks(
        evaluator,
        aera_root,
        field_id="Conference_Venue",
        field_human_desc="specific venue",
        extracted_value=extracted.venue,
        expected_values=GROUND_TRUTH["venue"],
        sources=sources,
        node_desc_from_rubric="Correctly identify Los Angeles Convention Center as the venue",
        match_tolerance_instruction=tolerance_instruction_for_field("venue"),
        url_instruction=url_instruction_for_field("venue"),
    )

    # Start Date
    await _add_text_field_checks(
        evaluator,
        aera_root,
        field_id="Conference_Start_Date",
        field_human_desc="conference start date",
        extracted_value=extracted.start_date,
        expected_values=GROUND_TRUTH["start_date"],
        sources=sources,
        node_desc_from_rubric="Correctly identify Wednesday, April 8, 2026 as the conference start date",
        match_tolerance_instruction=tolerance_instruction_for_field("start_date"),
        url_instruction=url_instruction_for_field("start_date"),
    )

    # End Date
    await _add_text_field_checks(
        evaluator,
        aera_root,
        field_id="Conference_End_Date",
        field_human_desc="conference end date",
        extracted_value=extracted.end_date,
        expected_values=GROUND_TRUTH["end_date"],
        sources=sources,
        node_desc_from_rubric="Correctly identify Sunday, April 12, 2026 as the conference end date",
        match_tolerance_instruction=tolerance_instruction_for_field("end_date"),
        url_instruction=url_instruction_for_field("end_date"),
    )

    # Presidential Address Speaker
    await _add_text_field_checks(
        evaluator,
        aera_root,
        field_id="Presidential_Address_Speaker",
        field_human_desc="speaker delivering the Presidential Address",
        extracted_value=extracted.presidential_address_speaker,
        expected_values=GROUND_TRUTH["presidential_address_speaker"],
        sources=sources,
        node_desc_from_rubric="Correctly identify Maisha T. Winn as the speaker delivering the Presidential Address",
        match_tolerance_instruction=tolerance_instruction_for_field("presidential_address_speaker"),
        url_instruction=url_instruction_for_field("presidential_address_speaker"),
    )

    # Presidential Address Date & Time
    await _add_text_field_checks(
        evaluator,
        aera_root,
        field_id="Presidential_Address_Date_Time",
        field_human_desc="scheduled date and time for the Presidential Address",
        extracted_value=extracted.presidential_address_date_time,
        expected_values=GROUND_TRUTH["presidential_address_date_time"],
        sources=sources,
        node_desc_from_rubric="Correctly identify Friday, April 10, 5:45 pm to 7:15 pm (Pacific Time) as the Presidential Address schedule",
        match_tolerance_instruction=tolerance_instruction_for_field("presidential_address_date_time"),
        url_instruction=url_instruction_for_field("presidential_address_date_time"),
    )

    # Opening Plenary Title
    await _add_text_field_checks(
        evaluator,
        aera_root,
        field_id="Opening_Plenary_Title",
        field_human_desc="title of the Opening Plenary",
        extracted_value=extracted.opening_plenary_title,
        expected_values=GROUND_TRUTH["opening_plenary_title"],
        sources=sources,
        node_desc_from_rubric="Correctly identify 'Holding Fast to Histories, Holding Fast to Dreams' as the title of the 2026 Opening Plenary",
        match_tolerance_instruction=tolerance_instruction_for_field("opening_plenary_title"),
        url_instruction=url_instruction_for_field("opening_plenary_title"),
    )

    # Distinguished Lecture Speaker
    await _add_text_field_checks(
        evaluator,
        aera_root,
        field_id="Distinguished_Lecture_Speaker",
        field_human_desc="speaker delivering the AERA Distinguished Lecture",
        extracted_value=extracted.distinguished_lecture_speaker,
        expected_values=GROUND_TRUTH["distinguished_lecture_speaker"],
        sources=sources,
        node_desc_from_rubric="Correctly identify Bryan Brayboy as the speaker delivering the AERA Distinguished Lecture",
        match_tolerance_instruction=tolerance_instruction_for_field("distinguished_lecture_speaker"),
        url_instruction=url_instruction_for_field("distinguished_lecture_speaker"),
    )

    # Wallace Foundation Distinguished Lecture Speaker
    await _add_text_field_checks(
        evaluator,
        aera_root,
        field_id="Wallace_Lecture_Speaker",
        field_human_desc="speaker delivering the Wallace Foundation Distinguished Lecture",
        extracted_value=extracted.wallace_lecture_speaker,
        expected_values=GROUND_TRUTH["wallace_lecture_speaker"],
        sources=sources,
        node_desc_from_rubric="Correctly identify Bianca Baldridge as the speaker delivering the Wallace Foundation Distinguished Lecture",
        match_tolerance_instruction=tolerance_instruction_for_field("wallace_lecture_speaker"),
        url_instruction=url_instruction_for_field("wallace_lecture_speaker"),
    )

    # Session count range (approximate)
    await _add_text_field_checks(
        evaluator,
        aera_root,
        field_id="Session_Count_Range",
        field_human_desc="approximate number of sessions (e.g., 'more than 2,500 sessions')",
        extracted_value=extracted.session_count_text,
        expected_values=GROUND_TRUTH["session_count_range"],
        sources=sources,
        node_desc_from_rubric="Correctly identify that the conference features more than 2,500 sessions",
        match_tolerance_instruction=tolerance_instruction_for_field("session_count_text"),
        url_instruction=url_instruction_for_field("session_count_text"),
    )

    # Time zone
    await _add_text_field_checks(
        evaluator,
        aera_root,
        field_id="Time_Zone",
        field_human_desc="time zone for all conference times",
        extracted_value=extracted.time_zone,
        expected_values=GROUND_TRUTH["time_zone"],
        sources=sources,
        node_desc_from_rubric="Correctly identify Pacific Time as the time zone for all conference times",
        match_tolerance_instruction=tolerance_instruction_for_field("time_zone"),
        url_instruction=url_instruction_for_field("time_zone"),
    )

    # Presenter Registration Deadline
    await _add_text_field_checks(
        evaluator,
        aera_root,
        field_id="Presenter_Registration_Deadline",
        field_human_desc="deadline for presenting authors to complete registration",
        extracted_value=extracted.presenter_registration_deadline,
        expected_values=GROUND_TRUTH["presenter_registration_deadline"],
        sources=sources,
        node_desc_from_rubric="Correctly identify February 13, 2026 as the deadline for presenter registration",
        match_tolerance_instruction=tolerance_instruction_for_field("presenter_registration_deadline"),
        url_instruction=url_instruction_for_field("presenter_registration_deadline"),
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
    Evaluate an answer for AERA 2026 Annual Meeting comprehensive details.
    """
    # Initialize evaluator
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

    # Extract structured details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_aera_details(),
        template_class=AERADetails,
        extraction_name="aera_2026_extracted_details",
    )

    # Add ground-truth info for transparency
    evaluator.add_ground_truth(
        {
            "expected_values": GROUND_TRUTH,
            "notes": "Canonical expected values based on rubric; minor formatting variants are acceptable.",
        },
        gt_type="ground_truth",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()