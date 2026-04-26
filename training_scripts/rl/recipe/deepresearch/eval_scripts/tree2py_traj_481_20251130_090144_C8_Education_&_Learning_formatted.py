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
TASK_ID = "bigten_2024_enrollment"
TASK_DESCRIPTION = (
    "Identify 4 universities that are members of the Big Ten Conference as of the 2024-25 academic year, "
    "where each university meets the following specific criteria based on their fall 2024 undergraduate enrollment numbers:\n\n"
    "University 1: Located in Pennsylvania with more than 40,000 undergraduate students enrolled in fall 2024.\n\n"
    "University 2: Located in Maryland, joined the Big Ten Conference in 2014, and has between 30,000 and 35,000 undergraduate students enrolled in fall 2024.\n\n"
    "University 3: Located in Ohio with more than 45,000 undergraduate students enrolled in fall 2024.\n\n"
    "University 4: Located in Illinois with between 35,000 and 40,000 undergraduate students enrolled in fall 2024.\n\n"
    "For each university, provide the university name and a reference URL that verifies the information."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityExtractionItem(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    claimed_state: Optional[str] = None
    claimed_joined_year: Optional[str] = None
    claimed_undergrad_enrollment_fall_2024: Optional[str] = None


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityExtractionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return (
        "Extract up to the first four universities presented in the answer that the author intends to satisfy the task. "
        "For each university, extract:\n"
        "1. name: The university's name.\n"
        "2. reference_urls: An array of all URLs explicitly provided in the answer that are intended to verify this university's facts (location, Big Ten membership, joined year if mentioned, Fall 2024 undergraduate enrollment). Extract only URLs; do not infer.\n"
        "3. claimed_state: The state mentioned for the university (if any) in the answer; otherwise null.\n"
        "4. claimed_joined_year: The year the university joined the Big Ten Conference as stated in the answer (if any); otherwise null.\n"
        "5. claimed_undergrad_enrollment_fall_2024: The Fall 2024 undergraduate enrollment as stated in the answer (number or phrase, if any); otherwise null.\n\n"
        "Return a JSON object with a 'universities' array of up to 4 items. "
        "If any field is missing for an item, set it to null (or empty array for URLs). "
        "Only include URLs that appear explicitly in the answer; if a URL is missing a protocol, prepend http://."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _pad_to_four(items: List[UniversityExtractionItem]) -> List[UniversityExtractionItem]:
    items = items[:4]
    while len(items) < 4:
        items.append(UniversityExtractionItem())
    return items


def _safe_name(item: UniversityExtractionItem, index: int) -> str:
    return item.name.strip() if item.name else f"University #{index}"


def _urls(item: UniversityExtractionItem) -> List[str]:
    return [u for u in item.reference_urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification logic per university                                           #
# --------------------------------------------------------------------------- #
async def verify_university_generic(
    evaluator: Evaluator,
    parent_node,
    item: UniversityExtractionItem,
    index: int,
    *,
    state: str,
    enrollment_kind: str,
    enrollment_lower: Optional[int] = None,
    enrollment_upper: Optional[int] = None,
    membership_2024_25: bool = True,
    joined_year: Optional[int] = None,
) -> None:
    """
    Build verification subtree for one university with provided constraints.
    """
    uni_num = index  # 1-based index expected by rubric
    uni_name = _safe_name(item, uni_num)
    srcs = _urls(item)

    # Parent node for this university
    uni_node = evaluator.add_parallel(
        id=f"University_{uni_num}",
        desc=(
            f"University {uni_num} satisfies constraints: {state}; Big Ten (2024-25)"
            + (f"; joined in {joined_year}" if joined_year else "")
            + (
                f"; Fall 2024 undergrad enrollment "
                + (f"> {enrollment_lower}" if enrollment_kind == "gt" and enrollment_lower is not None else "")
                + (f"between {enrollment_lower} and {enrollment_upper}" if enrollment_kind == "between" else "")
            )
            + "; includes reference URL."
        ),
        parent=parent_node,
        critical=False,
    )

    # Critical existence checks
    name_provided = evaluator.add_custom_node(
        result=bool(item.name and item.name.strip()),
        id=f"U{uni_num}_Name_Provided",
        desc=f"University {uni_num} name is provided.",
        parent=uni_node,
        critical=True,
    )

    urls_provided = evaluator.add_custom_node(
        result=bool(srcs),
        id=f"U{uni_num}_Reference_URL_Provided",
        desc=f"At least one reference URL is provided for University {uni_num}.",
        parent=uni_node,
        critical=True,
    )

    # Location verification
    loc_node = evaluator.add_leaf(
        id=f"U{uni_num}_Located_in_{state.replace(' ', '_')}",
        desc=f"University {uni_num} is located in {state} (claim is supported by the provided reference URL).",
        parent=uni_node,
        critical=True,
    )
    loc_claim = f"The university {uni_name} is located in {state}."
    loc_instruction = (
        "Verify the university's location (state) using the provided URL(s). "
        "Use the official/main campus location. If the page references a multi-campus system, "
        "confirm the flagship campus's state as appropriate. Do not rely on your own knowledge."
    )

    # Big Ten membership as of 2024-25
    bt_node = evaluator.add_leaf(
        id=f"U{uni_num}_BigTen_Member_2024_25",
        desc=f"University {uni_num} is a Big Ten Conference member as of the 2024-25 academic year (claim is supported by the provided reference URL).",
        parent=uni_node,
        critical=True,
    )
    bt_claim = f"{uni_name} is a Big Ten Conference member for the 2024-25 academic year."
    bt_instruction = (
        "Confirm that the university is a Big Ten member as of the 2024-25 academic year. "
        "The provided page should explicitly indicate Big Ten membership or list the institution among Big Ten members, "
        "reflecting the conference expansion status for 2024-25."
    )

    # Joined year (only for University 2 or if joined_year provided)
    joined_node = None
    joined_claim = None
    joined_instruction = None
    if joined_year is not None:
        joined_node = evaluator.add_leaf(
            id=f"U{uni_num}_Joined_BigTen_in_{joined_year}",
            desc=f"University {uni_num} joined the Big Ten Conference in {joined_year} (claim is supported by the provided reference URL).",
            parent=uni_node,
            critical=True,
        )
        joined_claim = f"{uni_name} joined the Big Ten Conference in {joined_year}."
        joined_instruction = (
            "Verify the year this university joined the Big Ten Conference. "
            "The page should explicitly state the joining year or provide a credible announcement/reference confirming it."
        )

    # Enrollment verification
    if enrollment_kind == "gt" and enrollment_lower is not None:
        enroll_desc_id = f"U{uni_num}_Fall2024_Undergrad_Enrollment_GT_{enrollment_lower}"
        enroll_desc_text = (
            f"University {uni_num} fall 2024 undergraduate enrollment is > {enrollment_lower} "
            "(claim is supported by the provided reference URL)."
        )
        enroll_claim = f"In Fall 2024, {uni_name} had more than {enrollment_lower} undergraduate students enrolled."
    elif enrollment_kind == "between" and enrollment_lower is not None and enrollment_upper is not None:
        enroll_desc_id = f"U{uni_num}_Fall2024_Undergrad_Enrollment_{enrollment_lower}_{enrollment_upper}"
        enroll_desc_text = (
            f"University {uni_num} fall 2024 undergraduate enrollment is between {enrollment_lower} and {enrollment_upper} "
            "(claim is supported by the provided reference URL)."
        )
        enroll_claim = (
            f"In Fall 2024, {uni_name} had between {enrollment_lower} and {enrollment_upper} undergraduate students enrolled."
        )
    else:
        # Fallback (should not happen with our fixed constraints)
        enroll_desc_id = f"U{uni_num}_Fall2024_Undergrad_Enrollment"
        enroll_desc_text = (
            f"University {uni_num} fall 2024 undergraduate enrollment matches the specified constraint "
            "(claim is supported by the provided reference URL)."
        )
        enroll_claim = f"In Fall 2024, {uni_name} had undergraduate enrollment that satisfies the specified constraint."

    enroll_node = evaluator.add_leaf(
        id=enroll_desc_id,
        desc=enroll_desc_text,
        parent=uni_node,
        critical=True,
    )
    enroll_instruction = (
        "Verify undergraduate (not total student) enrollment for Fall 2024. "
        "If a precise number is shown, interpret whether it satisfies the stated threshold/range. "
        "Allow reasonable rounding (e.g., 34,999 vs. ~35,000). The source should clearly indicate 'Fall 2024' and 'undergraduate'."
    )

    # Prepare batch verifications; automatic preconditions will skip if critical checks failed
    claims_batch: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = [
        (loc_claim, srcs, loc_node, loc_instruction),
        (bt_claim, srcs, bt_node, bt_instruction),
        (enroll_claim, srcs, enroll_node, enroll_instruction),
    ]
    if joined_node is not None and joined_claim is not None:
        claims_batch.append((joined_claim, srcs, joined_node, joined_instruction))

    await evaluator.batch_verify(claims_batch)


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Big Ten 2024-25 universities with enrollment constraints task.
    """
    # Initialize evaluator with a parallel root (non-critical root by framework design)
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

    # Extract universities and their reference URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    uni_items = _pad_to_four(extracted.universities or [])

    # Add a constraints summary for transparency
    evaluator.add_custom_info(
        info={
            "U1": {"state": "Pennsylvania", "enrollment": "> 40000", "joined_year": None},
            "U2": {"state": "Maryland", "enrollment": "30000–35000", "joined_year": 2014},
            "U3": {"state": "Ohio", "enrollment": "> 45000", "joined_year": None},
            "U4": {"state": "Illinois", "enrollment": "35000–40000", "joined_year": None},
            "membership_scope": "Big Ten member as of 2024-25 academic year",
            "enrollment_scope": "Undergraduate only, Fall 2024",
        },
        info_type="constraints",
        info_name="task_constraints",
    )

    # Build four university verification subtrees
    await verify_university_generic(
        evaluator,
        root,
        uni_items[0],
        1,
        state="Pennsylvania",
        enrollment_kind="gt",
        enrollment_lower=40000,
        membership_2024_25=True,
        joined_year=None,
    )
    await verify_university_generic(
        evaluator,
        root,
        uni_items[1],
        2,
        state="Maryland",
        enrollment_kind="between",
        enrollment_lower=30000,
        enrollment_upper=35000,
        membership_2024_25=True,
        joined_year=2014,
    )
    await verify_university_generic(
        evaluator,
        root,
        uni_items[2],
        3,
        state="Ohio",
        enrollment_kind="gt",
        enrollment_lower=45000,
        membership_2024_25=True,
        joined_year=None,
    )
    await verify_university_generic(
        evaluator,
        root,
        uni_items[3],
        4,
        state="Illinois",
        enrollment_kind="between",
        enrollment_lower=35000,
        enrollment_upper=40000,
        membership_2024_25=True,
        joined_year=None,
    )

    # Return structured evaluation summary
    return evaluator.get_summary()