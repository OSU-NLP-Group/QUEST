import asyncio
import logging
from typing import Any, List, Optional, Dict
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cabinet_secretaries_2025_2026"
TASK_DESCRIPTION = (
    "Identify four Cabinet-level department secretaries from the current administration who were confirmed by the "
    "United States Senate between January 20, 2025, and February 26, 2026, and who each received at least 60 votes "
    "in favor during their Senate confirmation vote. For each of the four secretaries, provide: (1) The official title "
    "of the Cabinet position (e.g., Secretary of State, Secretary of Defense); (2) The full name of the executive "
    "department they head; (3) The full name of the individual serving in the position; (4) The exact Senate "
    "confirmation vote breakdown (in Yea-Nay format); (5) The specific date when the Senate confirmed the nominee; "
    "(6) The official Senate roll call vote number; (7) A direct URL to an official U.S. government source (such as "
    "Senate.gov, the department's official website, or Congress.gov) that confirms the confirmation details. Note: "
    "The secretaries must head executive departments (not independent agencies), and all four must have received at "
    "least 60 votes in favor during their confirmation."
)

DATE_RANGE_START = "January 20, 2025"
DATE_RANGE_END = "February 26, 2026"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SecretaryItem(BaseModel):
    """One secretary entry extracted from the answer."""
    position_title: Optional[str] = None  # e.g., "Secretary of Defense"
    department_name: Optional[str] = None  # e.g., "Department of Defense"
    secretary_name: Optional[str] = None  # e.g., "Jane Doe"
    vote_details: Optional[str] = None  # e.g., "78-20" (Yea-Nay)
    confirmation_date: Optional[str] = None  # e.g., "February 3, 2025"
    roll_call_number: Optional[str] = None  # e.g., "Roll Call Vote #45"
    reference_url: Optional[str] = None  # official gov source confirming confirmation details
    additional_urls: List[str] = Field(default_factory=list)  # other official gov URLs (dept site, Congress.gov, etc.)


class SecretariesExtraction(BaseModel):
    """List of secretaries extracted from the answer."""
    secretaries: List[SecretaryItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_secretaries() -> str:
    return """
    Extract up to four Cabinet-level department secretaries from the answer text, each with the following fields:

    Required per secretary:
    1. position_title: The official title of the Cabinet position (e.g., "Secretary of State", "Secretary of Defense").
    2. department_name: The full name of the executive department they head (e.g., "Department of State").
    3. secretary_name: The full name of the individual serving in the position.
    4. vote_details: The exact Senate confirmation vote breakdown in "Yea-Nay" format (e.g., "78-20"). If not provided, set to null.
    5. confirmation_date: The specific date when the Senate confirmed the nominee (as presented in the answer; keep original formatting).
    6. roll_call_number: The official Senate roll call vote number, if provided (e.g., "Roll Call Vote 45"); otherwise null.
    7. reference_url: A single direct URL to an official U.S. government source (e.g., Senate.gov, Congress.gov, or a .gov/.mil department site) that confirms the confirmation details. If none is provided, set to null.
    8. additional_urls: A list of any other official U.S. government URLs mentioned for this secretary (e.g., department press release about being sworn in, current leadership page). If none are provided, return an empty list.

    Rules:
    - Extract information exactly as presented in the answer; do not invent or infer missing data.
    - Only include URLs that are explicitly present in the answer.
    - If more than four secretaries are mentioned, include only the first four in the extracted list.
    - If any field is missing for a secretary, set it to null (or an empty list for additional_urls).

    Return a JSON object with a single field "secretaries" which is an array of secretary objects
    following the schema above.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_official_us_gov_url(url: Optional[str]) -> bool:
    """
    Basic check whether a URL points to an official U.S. government domain.
    Accepts *.gov and *.mil domains.
    """
    if not url:
        return False
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.endswith(".gov") or netloc.endswith(".mil")
    except Exception:
        return False


def combined_sources(item: SecretaryItem) -> List[str]:
    """
    Combine the primary reference URL with any additional URLs.
    """
    urls: List[str] = []
    if item.reference_url:
        urls.append(item.reference_url)
    if item.additional_urls:
        urls.extend([u for u in item.additional_urls if isinstance(u, str) and u.strip() != ""])
    return urls


# --------------------------------------------------------------------------- #
# Verification for one secretary                                              #
# --------------------------------------------------------------------------- #
async def verify_secretary(
    evaluator: Evaluator,
    parent_node,
    item: SecretaryItem,
    idx: int
) -> None:
    """
    Build verification sub-tree and run checks for one secretary.
    """
    sec_node = evaluator.add_parallel(
        id=f"Secretary_{idx+1}",
        desc=f"{['First','Second','Third','Fourth'][idx]} Cabinet secretary meeting all criteria",
        parent=parent_node,
        critical=False  # allow partial scoring across items
    )

    # Leaf: Reference URL presence and official domain (critical prerequisite for other verifications)
    ref_url_node = evaluator.add_custom_node(
        result=(item.reference_url is not None and is_official_us_gov_url(item.reference_url)),
        id=f"S{idx+1}_Reference_URL",
        desc="Provides a URL to an official government source (Senate.gov, department website, or Congress.gov) confirming the confirmation details",
        parent=sec_node,
        critical=True
    )

    # Leaf: Position Identification (verify the page corresponds to this Cabinet secretary position)
    pos_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Position_Identification",
        desc="Provides the official title of a Cabinet-level department secretary position (e.g., Secretary of State, Secretary of Defense, etc.)",
        parent=sec_node,
        critical=True
    )
    pos_claim = (
        f"This official page concerns the nomination/confirmation to the position '{item.position_title}'. "
        f"If the page states 'to be {item.position_title}' or otherwise clearly indicates the role, consider it supported."
    )
    await evaluator.verify(
        claim=pos_claim,
        node=pos_leaf,
        sources=combined_sources(item),
        additional_instruction="Focus on whether the page explicitly identifies the position title as presented."
    )

    # Leaf: Department Name presence (critical presence check; verification of exec department is separate)
    dept_leaf = evaluator.add_custom_node(
        result=(item.department_name is not None and item.department_name.strip() != ""),
        id=f"S{idx+1}_Department_Name",
        desc="Provides the full name of the executive department headed by this secretary",
        parent=sec_node,
        critical=True
    )

    # Leaf: Executive Department verification (source-grounded)
    exec_dept_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Executive_Department_Verification",
        desc="Confirms that the position heads an executive department (not an independent agency such as EPA, NASA, etc.)",
        parent=sec_node,
        critical=True
    )
    exec_dept_claim = (
        f"The position '{item.position_title}' heads an executive department of the United States Government "
        f"(not an independent agency)."
    )
    await evaluator.verify(
        claim=exec_dept_claim,
        node=exec_dept_leaf,
        sources=combined_sources(item),
        additional_instruction=(
            "Treat positions styled 'Secretary of [Department]' as heading executive departments (e.g., Department of State, "
            "Defense, Treasury, etc.). If the page indicates an independent agency (e.g., EPA Administrator, NASA Administrator), "
            "this claim is not supported."
        )
    )

    # Leaf: Secretary Name (source-grounded)
    name_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Secretary_Name",
        desc="Provides the full name of the individual serving in this position",
        parent=sec_node,
        critical=True
    )
    name_claim = (
        f"The individual confirmed for the position '{item.position_title}' is '{item.secretary_name}'. "
        f"Accept reasonable name variants (middle initials, capitalization)."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=combined_sources(item),
        additional_instruction="Verify the confirmed nominee's full name on the page."
    )

    # Leaf: Vote Threshold ≥ 60 Yeas (source-grounded)
    threshold_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Vote_Threshold",
        desc="The confirmation received at least 60 votes in favor",
        parent=sec_node,
        critical=True
    )
    threshold_claim = "The Senate confirmation vote recorded at least 60 Yea votes."
    await evaluator.verify(
        claim=threshold_claim,
        node=threshold_leaf,
        sources=combined_sources(item),
        additional_instruction="Check the Yea count on the official roll call page."
    )

    # Leaf: Bipartisan Support (source-grounded)
    bipartisan_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Bipartisan_Support",
        desc="The confirmation vote included votes from both Republican and Democratic senators (bipartisan support)",
        parent=sec_node,
        critical=True
    )
    bipartisan_claim = (
        "The Yea votes included senators from both the Republican and Democratic parties (bipartisan support)."
    )
    await evaluator.verify(
        claim=bipartisan_claim,
        node=bipartisan_leaf,
        sources=combined_sources(item),
        additional_instruction="Use party breakdown on the roll call page if available; names may also indicate party."
    )

    # Leaf: Confirmation Date supported by source (source-grounded)
    conf_date_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Confirmation_Date",
        desc=f"Provides the specific date when the Senate confirmed this nominee (must be between {DATE_RANGE_START} and {DATE_RANGE_END})",
        parent=sec_node,
        critical=True
    )
    conf_date_claim = (
        f"The Senate confirmed the nominee on '{item.confirmation_date}'. Match the date shown on the official page."
    )
    await evaluator.verify(
        claim=conf_date_claim,
        node=conf_date_leaf,
        sources=combined_sources(item),
        additional_instruction="Confirm the specific calendar date of the confirmation on the official source."
    )

    # Leaf: Confirmation Date in range (logical check without source)
    conf_date_range_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Confirmation_Date_In_Range",
        desc=f"Confirmation date falls between {DATE_RANGE_START} and {DATE_RANGE_END} (inclusive)",
        parent=sec_node,
        critical=True
    )
    conf_date_range_claim = (
        f"The stated confirmation date '{item.confirmation_date}' falls between {DATE_RANGE_START} and {DATE_RANGE_END}, inclusive."
    )
    await evaluator.verify(
        claim=conf_date_range_claim,
        node=conf_date_range_leaf,
        sources=None,
        additional_instruction=(
            "Interpret typical date formats (e.g., 'February 3, 2025', '2025-02-03'). If the date clearly lies outside the "
            "range, mark incorrect."
        )
    )

    # Leaf: Vote Details exact Yea-Nay (source-grounded)
    vote_details_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Vote_Details",
        desc="Provides the complete vote breakdown (Yea-Nay format)",
        parent=sec_node,
        critical=True
    )
    vote_details_claim = (
        f"The exact vote breakdown was '{item.vote_details}' in Yea-Nay format. "
        f"Other categories like Present/Not Voting should be ignored for this comparison."
    )
    await evaluator.verify(
        claim=vote_details_claim,
        node=vote_details_leaf,
        sources=combined_sources(item),
        additional_instruction="Compare the Yea and Nay counts only."
    )

    # Leaf: Roll Call number (source-grounded)
    roll_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Roll_Call_Number",
        desc="Provides the official Senate roll call vote number",
        parent=sec_node,
        critical=True
    )
    roll_claim = f"The official Senate roll call vote number associated with this confirmation is '{item.roll_call_number}'."
    await evaluator.verify(
        claim=roll_claim,
        node=roll_leaf,
        sources=combined_sources(item),
        additional_instruction="Check the roll call number designation on the page (e.g., 'Roll Call Vote #45')."
    )

    # Leaf: Currently Serving (source-grounded; set non-critical to avoid over-penalizing if not provided)
    curr_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Currently_Serving",
        desc="Confirms that the individual is currently serving in the position (not resigned or removed)",
        parent=sec_node,
        critical=False  # adjusted to non-critical; not explicitly required by the original task statement
    )
    curr_claim = (
        f"As of today, '{item.secretary_name}' is currently serving as '{item.position_title}' heading the '{item.department_name}'."
    )
    await evaluator.verify(
        claim=curr_claim,
        node=curr_leaf,
        sources=combined_sources(item),
        additional_instruction=(
            "Confirm current service using official department leadership page or recent official statements. "
            "If no such confirmation appears in provided government URLs, mark not supported."
        )
    )

    # Leaf: Oath sworn (source-grounded; set non-critical to avoid over-penalizing)
    oath_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Oath_Sworn_In",
        desc="Confirms that the individual has taken the oath of office and been officially sworn in",
        parent=sec_node,
        critical=False  # adjusted to non-critical; not explicitly required by the original task statement
    )
    oath_claim = f"'{item.secretary_name}' has taken the oath of office and been officially sworn in as '{item.position_title}'."
    await evaluator.verify(
        claim=oath_claim,
        node=oath_leaf,
        sources=combined_sources(item),
        additional_instruction="Department press releases or official statements often confirm swearing-in."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Cabinet secretaries confirmation task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # independent secretaries aggregated in parallel
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

    # Extract secretaries
    extracted = await evaluator.extract(
        prompt=prompt_extract_secretaries(),
        template_class=SecretariesExtraction,
        extraction_name="secretaries_extraction"
    )

    # Add ground truth-like constraints information (for context in summary)
    evaluator.add_ground_truth({
        "date_range_required": {"start": DATE_RANGE_START, "end": DATE_RANGE_END},
        "vote_threshold_required": "At least 60 Yea votes",
        "source_requirement": "Official U.S. government URL (.gov or .mil), e.g., Senate.gov, Congress.gov, department sites"
    }, gt_type="task_constraints")

    # Take up to first 4 secretaries; pad if fewer
    secretaries: List[SecretaryItem] = list(extracted.secretaries[:4])
    while len(secretaries) < 4:
        secretaries.append(SecretaryItem())

    # Build rubric root node (matches JSON root description)
    task_node = evaluator.add_parallel(
        id="Cabinet_Secretaries_Task",
        desc="Identify four Cabinet-level department secretaries confirmed by the Senate between Jan 20, 2025 and Feb 26, 2026 with ≥60 Yea votes",
        parent=root,
        critical=False
    )

    # Verify each secretary
    for i, item in enumerate(secretaries):
        await verify_secretary(evaluator, task_node, item, i)

    # Return evaluation summary
    return evaluator.get_summary()