import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "robotics_conf_2025_us"
TASK_DESCRIPTION = (
    "A graduate student needs to submit a robotics research paper to an academic conference with the following requirements: "
    "(1) The conference must be held in the United States; "
    "(2) The conference must take place between April 1, 2025 and June 30, 2025; "
    "(3) The venue must be a convention center or similar large-scale international conference facility; "
    "(4) The early bird registration fee for student members must be less than $350; "
    "(5) The conference must accept papers in computer science or robotics; "
    "(6) The conference must have a defined paper submission format (IEEE, ACM, or similar). "
    "Identify one conference that meets all these criteria. Provide: "
    "(a) the full conference name, (b) the city and state location, (c) the venue name, "
    "(d) the complete conference dates, (e) the early bird registration fee for student members along with the membership category, "
    "and (f) reference URLs supporting your information."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConferenceInfo(BaseModel):
    """Structured extraction for a single conference (first one in the answer)."""
    conference_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    venue_name: Optional[str] = None
    # Optional venue type descriptor as stated in the answer (e.g., 'Convention Center', 'Conference Center', etc.)
    venue_type: Optional[str] = None

    # Dates as strings (keep flexible formats)
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Registration fee info for student members, early-bird
    student_membership_category: Optional[str] = None  # e.g., "IEEE Student Member", "ACM Student Member"
    early_bird_student_member_fee: Optional[str] = None  # e.g., "$299", "USD 320", or similar string

    # Scope/domain and submission format
    accepted_domains: List[str] = Field(default_factory=list)  # e.g., ["Computer Science", "Robotics", "AI"]
    paper_submission_format: Optional[str] = None  # e.g., "IEEE", "ACM", "Springer LNCS", etc.

    # References supporting the information (URLs explicitly cited in the answer)
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conference() -> str:
    return """
    Extract exactly ONE conference (the first one if multiple are mentioned) that the answer proposes for submission.
    Return a JSON object with the following fields, using values exactly as they appear in the answer:

    - conference_name: The full official conference name.
    - city: The city where the conference is held.
    - state: The U.S. state where the conference is held.
    - venue_name: The venue/facility name where the conference takes place.
    - venue_type: The venue type descriptor if present (e.g., "Convention Center", "Conference Center", "Expo Center", "Congress Center", etc.). If not explicitly stated, return null.
    - start_date: The conference start date (string; keep the format used in the answer).
    - end_date: The conference end date (string; keep the format used in the answer).
    - student_membership_category: The membership category relevant to student early-bird registration (e.g., "IEEE Student Member", "ACM Student Member"). If not provided, return null.
    - early_bird_student_member_fee: The early-bird registration fee for student members as a string exactly as in the answer (e.g., "$299", "USD 320", etc.). If not provided, return null.
    - accepted_domains: An array of subject domains explicitly stated to be accepted by the conference (e.g., ["Computer Science", "Robotics", "AI"]). If not provided, return an empty array.
    - paper_submission_format: The paper submission format name or descriptor (e.g., "IEEE", "ACM", "Springer LNCS", "Elsevier", etc.). If not provided, return null.
    - reference_urls: An array of all URLs explicitly cited in the answer that support the conference info (include CFP pages, registration pages, venue pages, etc.). If none are provided, return an empty array.

    IMPORTANT:
    - Extract only what is explicitly present in the answer text; do not invent or infer values.
    - If an item is missing, set its JSON field to null (or empty array for lists).
    - If multiple conferences are provided, return information for the first one only.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _join_urls(urls: List[str]) -> List[str]:
    """Return the list as-is; helper to guard against None."""
    return urls or []


# --------------------------------------------------------------------------- #
# Build verification tree and run checks                                      #
# --------------------------------------------------------------------------- #
async def verify_conference(
    evaluator: Evaluator,
    parent_node,
    info: ConferenceInfo,
) -> None:
    """
    Construct the verification tree for the selected conference and run evidence-based checks.
    All nodes under the top-level are critical, matching rubric semantics.
    """

    # Top-level task node (critical, parallel aggregation)
    task_node = evaluator.add_parallel(
        id="Conference_Meeting_All_Criteria",
        desc="The answer identifies one conference and demonstrates it meets all specified criteria, providing all requested fields and supporting references.",
        parent=parent_node,
        critical=True,
    )

    # References existence (critical leaf at top-level; also used as precondition for evidence-based checks)
    references_exist = bool(info.reference_urls)
    references_node = evaluator.add_custom_node(
        result=references_exist,
        id="References_Provided",
        desc="Reference URL(s) are provided that support the stated information (e.g., name, location/venue, dates, and fees).",
        parent=task_node,
        critical=True,
    )

    # Conference identity (critical)
    identity_node = evaluator.add_parallel(
        id="Conference_Identity",
        desc="The conference is clearly identified.",
        parent=task_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(info.conference_name and info.conference_name.strip()),
        id="Conference_Name_Provided",
        desc="The full official conference name is provided.",
        parent=identity_node,
        critical=True,
    )

    # Location requirements (critical)
    location_node = evaluator.add_parallel(
        id="Location_Requirements",
        desc="The conference location and venue meet the geographic and facility requirements, and required location fields are provided.",
        parent=task_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(info.city and info.city.strip() and info.state and info.state.strip()),
        id="City_State_Provided",
        desc="The city and state are provided.",
        parent=location_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(info.venue_name and info.venue_name.strip()),
        id="Venue_Name_Provided",
        desc="The venue name is provided.",
        parent=location_node,
        critical=True,
    )

    # US location verification (critical leaf)
    us_loc_leaf = evaluator.add_leaf(
        id="US_Location",
        desc="The conference is held within the United States.",
        parent=location_node,
        critical=True,
    )
    loc_claim = f"The conference is held in {info.city}, {info.state}, United States."
    await evaluator.verify(
        claim=loc_claim,
        node=us_loc_leaf,
        sources=_join_urls(info.reference_urls),
        additional_instruction=(
            "Verify that the city and state are in the United States based on the referenced pages "
            "(e.g., conference website, venue page, or CFP/registration pages). Minor variations in formatting are acceptable."
        ),
        extra_prerequisites=[references_node],
    )

    # Venue type constraint (critical leaf)
    venue_type_leaf = evaluator.add_leaf(
        id="Venue_Type_Satisfies_Constraint",
        desc="The venue is a convention center or similar large-scale facility suitable for international conferences.",
        parent=location_node,
        critical=True,
    )
    venue_type_claim = (
        f"The venue '{info.venue_name}' is a convention center or a similar large-scale international conference facility."
    )
    await evaluator.verify(
        claim=venue_type_claim,
        node=venue_type_leaf,
        sources=_join_urls(info.reference_urls),
        additional_instruction=(
            "Check whether the venue is explicitly a 'Convention Center', 'Conference Center', 'Expo Center', 'Congress Center', "
            "or an equivalent large-scale international conference facility. Venue pages, conference venue info pages, or trusted listings "
            "that clearly indicate such a facility type are acceptable."
        ),
        extra_prerequisites=[references_node],
    )

    # Temporal requirements (critical)
    temporal_node = evaluator.add_parallel(
        id="Temporal_Requirements",
        desc="The conference dates are provided and fall within the required window.",
        parent=task_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(info.start_date and info.start_date.strip() and info.end_date and info.end_date.strip()),
        id="Conference_Dates_Provided",
        desc="The complete conference dates (start and end) are provided.",
        parent=temporal_node,
        critical=True,
    )

    # Within target period verification (critical leaf)
    within_period_leaf = evaluator.add_leaf(
        id="Within_Target_Period",
        desc="The conference takes place between April 1, 2025 and June 30, 2025.",
        parent=temporal_node,
        critical=True,
    )
    period_claim = (
        f"The conference runs from {info.start_date} to {info.end_date}, and these dates fall between April 1, 2025 and June 30, 2025 (inclusive)."
    )
    await evaluator.verify(
        claim=period_claim,
        node=within_period_leaf,
        sources=_join_urls(info.reference_urls),
        additional_instruction=(
            "Verify the event dates on the official conference website or reputable pages. "
            "Ensure the event dates (not submission deadlines) are within the window April 1, 2025 to June 30, 2025."
        ),
        extra_prerequisites=[references_node],
    )

    # Financial requirements (critical)
    financial_node = evaluator.add_parallel(
        id="Financial_Requirements",
        desc="The early bird student-member registration fee is provided and meets the budget constraint.",
        parent=task_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(info.early_bird_student_member_fee and info.early_bird_student_member_fee.strip() and info.student_membership_category and info.student_membership_category.strip()),
        id="Student_Early_Bird_Fee_And_Category_Provided",
        desc="The early bird registration fee for student members is provided along with the membership category.",
        parent=financial_node,
        critical=True,
    )

    fee_leaf = evaluator.add_leaf(
        id="Fee_Below_Threshold",
        desc="The early bird registration fee for student members is less than $350.",
        parent=financial_node,
        critical=True,
    )
    fee_claim = (
        f"The early-bird registration fee for {info.student_membership_category} is less than $350."
    )
    await evaluator.verify(
        claim=fee_claim,
        node=fee_leaf,
        sources=_join_urls(info.reference_urls),
        additional_instruction=(
            "Check registration/fees pages for 'Student Member' or equivalent early-bird category. "
            "Confirm the early-bird student member rate is < $350 (USD). If multiple currencies or taxes are present, use the main listed base rate."
        ),
        extra_prerequisites=[references_node],
    )

    # Domain and format requirements (critical)
    domain_node = evaluator.add_parallel(
        id="Domain_and_Format_Requirements",
        desc="The conference accepts relevant research and specifies a recognized submission format.",
        parent=task_node,
        critical=True,
    )

    # Domain acceptance (critical leaf)
    domain_leaf = evaluator.add_leaf(
        id="Domain_Match",
        desc="The conference accepts papers in computer science or robotics.",
        parent=domain_node,
        critical=True,
    )
    # Build a helpful claim using extracted domains if available
    if info.accepted_domains:
        dom_list = ", ".join(info.accepted_domains)
        domain_claim = (
            f"The conference call for papers includes topics in computer science or robotics. "
            f"Accepted domains mentioned include: {dom_list}."
        )
    else:
        domain_claim = "The conference accepts paper submissions in computer science or robotics."

    await evaluator.verify(
        claim=domain_claim,
        node=domain_leaf,
        sources=_join_urls(info.reference_urls),
        additional_instruction=(
            "Use the CFP/scope/topics page to check whether CS/Robotics research papers are explicitly within scope. "
            "Names like 'Computer Science', 'Robotics', 'Autonomous Systems', 'AI for Robotics', 'Mechatronics', etc., count as relevant."
        ),
        extra_prerequisites=[references_node],
    )

    # Paper format defined (critical leaf)
    format_leaf = evaluator.add_leaf(
        id="Paper_Format_Defined",
        desc="The conference has a clearly defined paper submission format (IEEE, ACM, or similar).",
        parent=domain_node,
        critical=True,
    )
    if info.paper_submission_format:
        format_claim = (
            f"The conference requires a clearly defined submission format and uses {info.paper_submission_format} or an equivalent recognized style."
        )
    else:
        format_claim = "The conference defines a recognized paper submission format such as IEEE, ACM, Springer LNCS, or a similar standard."

    await evaluator.verify(
        claim=format_claim,
        node=format_leaf,
        sources=_join_urls(info.reference_urls),
        additional_instruction=(
            "Check author guidelines or submission instructions for explicit format requirements (e.g., IEEE, ACM, Springer LNCS). "
            "Equivalent recognized styles are acceptable if clearly defined."
        ),
        extra_prerequisites=[references_node],
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
    Evaluate an answer for the 2025 U.S. robotics conference requirements task.
    """
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

    # Extract structured conference info from the answer
    conf_info = await evaluator.extract(
        prompt=prompt_extract_conference(),
        template_class=ConferenceInfo,
        extraction_name="conference_info",
    )

    # Build verification tree and run checks
    await verify_conference(evaluator, root, conf_info)

    # Return standardized evaluation summary
    return evaluator.get_summary()