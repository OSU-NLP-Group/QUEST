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
TASK_ID = "tx_superintendent_internal_promo_2024plus"
TASK_DESCRIPTION = (
    "Identify a superintendent of a Texas school district who was officially appointed to the "
    "superintendent position in 2024 or later and who previously served as either a deputy superintendent "
    "or assistant superintendent within that same district before their promotion. The individual must have "
    "joined the district before 2024. Provide: superintendent's full name, the district's full official name, "
    "the specific date of official appointment as superintendent, the title of their previous position within "
    "the same district, and a reference URL confirming this information."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class SuperintendentEntry(BaseModel):
    superintendent_full_name: Optional[str] = None
    district_full_official_name: Optional[str] = None
    superintendent_appointment_date: Optional[str] = None
    previous_position_title: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)

    # Optional / if-available fields (kept for potential custom info logging)
    previous_position_start_date: Optional[str] = None
    isd_status: Optional[str] = None
    career_progression_timeline: Optional[str] = None
    board_vote_info: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendent() -> str:
    return """
Extract the superintendent information provided in the answer.

Return a single JSON object with the following fields (use null if a field is missing; do not fabricate):
- superintendent_full_name: The superintendent's full name as stated in the answer.
- district_full_official_name: The full official name of the school district as stated in the answer (e.g., "XYZ Independent School District" or "XYZ ISD"; do not abbreviate further).
- superintendent_appointment_date: The specific date (as written in the answer) of the official appointment or board approval as superintendent (not "lone/sole finalist" selection; use the actual appointment/approval date).
- previous_position_title: The title of the individual's previous role within the same district (e.g., "Deputy Superintendent", "Assistant Superintendent of Curriculum").
- reference_urls: An array of all URLs cited in the answer that support the claims (board announcements, district press releases, news coverage, district website pages, etc.). Extract only the actual URLs mentioned in the answer.

Also extract the following optional fields if the answer includes them:
- previous_position_start_date: The date or year the individual began their previous position within the district.
- isd_status: Any explicit statement that the district is an "Independent School District (ISD)"; return the text if present (e.g., "Independent School District", "ISD").
- career_progression_timeline: Any narrative text in the answer describing the timeline of progression from the prior role to superintendent.
- board_vote_info: Any details of the board of trustees vote/decision process (e.g., vote date, unanimous decision).

Rules:
- Do not infer or invent. Only extract content explicitly present in the answer.
- For URLs, accept plain URLs or markdown links; collect all unique valid URLs.
- Keep dates as free-form strings exactly as written (e.g., "June 10, 2024", "2024-06-10", "6/10/2024").
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len([u for u in urls if _non_empty(u)]) > 0)


# --------------------------------------------------------------------------- #
# Build verification tree                                                     #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extracted: SuperintendentEntry,
) -> None:
    """
    Build the verification tree according to the rubric.
    Note: We focus the scoring tree on the two critical branches:
      1) Eligibility_Criteria (critical)
      2) Required_Output_Fields (critical)

    Optional 'if available' fields are recorded via custom info to avoid distorting
    final gating behavior; omission of those does not penalize the overall pass/fail.
    """

    # ---------------- Required Output Fields (critical; presence checks) ----------------
    required_node = evaluator.add_parallel(
        id="Required_Output_Fields",
        desc="All required fields requested by the question are provided.",
        parent=evaluator.root,
        critical=True
    )

    name_present = evaluator.add_custom_node(
        result=_non_empty(extracted.superintendent_full_name),
        id="Superintendent_Full_Name_Provided",
        desc="The superintendent's full name is provided.",
        parent=required_node,
        critical=True
    )

    district_present = evaluator.add_custom_node(
        result=_non_empty(extracted.district_full_official_name),
        id="District_Official_Name_Provided",
        desc="The full official name of the school district is provided.",
        parent=required_node,
        critical=True
    )

    appoint_date_present = evaluator.add_custom_node(
        result=_non_empty(extracted.superintendent_appointment_date),
        id="Official_Appointment_Date_Provided",
        desc="The specific date of the official appointment as superintendent is provided.",
        parent=required_node,
        critical=True
    )

    prev_title_present = evaluator.add_custom_node(
        result=_non_empty(extracted.previous_position_title),
        id="Previous_Position_Title_Provided",
        desc="The specific title of the previous position held within the same district is provided.",
        parent=required_node,
        critical=True
    )

    refs_present = evaluator.add_custom_node(
        result=_has_any_url(extracted.reference_urls),
        id="Reference_URL_Provided",
        desc="At least one valid reference URL is provided that supports the superintendent's appointment information.",
        parent=required_node,
        critical=True
    )

    # ---------------- Eligibility Criteria (critical; all constraints must hold) ----------------
    eligibility_node = evaluator.add_parallel(
        id="Eligibility_Criteria",
        desc="The selected individual/district satisfies all required eligibility constraints.",
        parent=evaluator.root,
        critical=True
    )

    # 1) District located in Texas
    district_tx_leaf = evaluator.add_leaf(
        id="District_Located_in_Texas",
        desc="The school district is located in Texas.",
        parent=eligibility_node,
        critical=True
    )
    district_name = extracted.district_full_official_name or ""
    await evaluator.verify(
        claim=f"The school district '{district_name}' is located in Texas.",
        node=district_tx_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Use the provided URL(s) to confirm the district is in Texas. "
            "Accept clear indications such as 'Texas', '.tx.us' domains, mentions of Texas education agencies, "
            "or explicit statements on official district/news pages that the district serves communities in Texas."
        ),
        extra_prerequisites=[district_present, refs_present]
    )

    # 2) Appointed as Superintendent in 2024 or later (and is the official appointment/board approval date)
    appointed_leaf = evaluator.add_leaf(
        id="Appointed_as_Superintendent_2024_or_Later",
        desc="The individual was officially appointed as superintendent in 2024 or later.",
        parent=eligibility_node,
        critical=True
    )
    sup_name = extracted.superintendent_full_name or ""
    appoint_date = extracted.superintendent_appointment_date or ""
    await evaluator.verify(
        claim=(
            f"On {appoint_date}, which is in 2024 or later, {sup_name} was officially appointed or approved by the board "
            f"as superintendent of {district_name}."
        ),
        node=appointed_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Confirm the event is an official appointment/board approval (not merely selection as 'lone/sole finalist'). "
            "The date must be in 2024 or later."
        ),
        extra_prerequisites=[name_present, district_present, appoint_date_present, refs_present]
    )

    # 3) Previously served as deputy or assistant within the same district (before promotion)
    prev_role_leaf = evaluator.add_leaf(
        id="Previously_Served_Deputy_or_Assistant_Same_District",
        desc="Before promotion, the individual served as a deputy superintendent or assistant superintendent within the same district.",
        parent=eligibility_node,
        critical=True
    )
    prev_title = extracted.previous_position_title or ""
    await evaluator.verify(
        claim=(
            f"Before their superintendent appointment, {sup_name} served within {district_name} as either a deputy superintendent "
            f"or an assistant superintendent. Their stated previous title was '{prev_title}'."
        ),
        node=prev_role_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Verify that the prior role in the same district is specifically 'Deputy Superintendent' or 'Assistant Superintendent' "
            "(including reasonable variants like 'Deputy Supt.' or 'Assistant Superintendent of XYZ'). "
            "Do NOT count 'Associate Superintendent' or 'Interim Superintendent' for this criterion."
        ),
        extra_prerequisites=[name_present, district_present, prev_title_present, refs_present]
    )

    # 4) Joined the district before 2024
    joined_before_leaf = evaluator.add_leaf(
        id="Joined_District_Before_2024",
        desc="The individual joined the district before 2024.",
        parent=eligibility_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{sup_name} joined {district_name} before the year 2024.",
        node=joined_before_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Look for explicit start dates/years with the district (e.g., started in 2022) or language clearly indicating tenure "
            "before 2024. The evidence must show they were employed by the district prior to 2024."
        ),
        extra_prerequisites=[name_present, district_present, refs_present]
    )

    # ---------------- Optional (if available) fields: add as custom info (non-scoring) ----------------
    optional_info = {
        "previous_position_start_date": extracted.previous_position_start_date,
        "isd_status": extracted.isd_status,
        "career_progression_timeline": extracted.career_progression_timeline,
        "board_vote_info": extracted.board_vote_info,
    }
    evaluator.add_custom_info(optional_info, info_type="optional_if_available", info_name="optional_fields_extracted")


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
    Evaluate an answer for the Texas superintendent internal-promotion task.
    """
    # Initialize evaluator (root is non-critical, parallel aggregation)
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
        default_model=model
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_superintendent(),
        template_class=SuperintendentEntry,
        extraction_name="superintendent_extraction"
    )

    # Record ground truth requirements (for context in the summary)
    evaluator.add_ground_truth({
        "constraints": {
            "state": "Texas",
            "appointment_year_min": 2024,
            "prior_role_required": ["Deputy Superintendent", "Assistant Superintendent"],
            "joined_before_year": 2024
        },
        "required_fields": [
            "superintendent_full_name",
            "district_full_official_name",
            "superintendent_appointment_date",
            "previous_position_title",
            "reference_urls"
        ]
    }, gt_type="task_requirements")

    # Build verification tree according to rubric (critical: eligibility + required fields)
    await build_verification_tree(evaluator, extracted)

    # Return final evaluation summary
    return evaluator.get_summary()