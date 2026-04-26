import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nc_motion_admission"
TASK_DESCRIPTION = (
    "An attorney who has been actively practicing law in New York for the past 5 years and passed the Uniform Bar Exam "
    "with a score of 268 is planning to relocate to North Carolina. They wish to gain admission to the North Carolina "
    "State Bar without retaking the bar examination. What are the specific requirements they must satisfy to be admitted "
    "on motion to practice law in North Carolina, and will their UBE score be accepted?"
)

NC_PRACTICE_CATEGORIES = [
    "law teaching",
    "government agency work",
    "military service",
    "in-house corporate practice",
    "service in a judicial court of record",
]


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class MotionAdmissionExtraction(BaseModel):
    # Reciprocity / Admission on Motion eligibility for NY attorneys
    reciprocity_statement: Optional[str] = None
    reciprocity_sources: List[str] = Field(default_factory=list)

    # Years-of-practice requirement and definition of "practice"
    practice_rule_text: Optional[str] = None  # e.g., "4 years of active practice out of the past 6 years"
    practice_required_years_rule: Optional[str] = None  # normalized short text if present (e.g., "4 of past 6")
    practice_categories_mentioned: List[str] = Field(default_factory=list)
    practice_sources: List[str] = Field(default_factory=list)

    # UBE transfer / acceptance
    nc_ube_jurisdiction_statement: Optional[str] = None
    ube_transfer_acceptance_statement: Optional[str] = None
    ube_portability_condition_text: Optional[str] = None  # e.g., "MEE, MPT, MBE in same jurisdiction & administration"
    nc_minimum_ube_score: Optional[str] = None  # numeric or text; we'll parse
    ube_sources: List[str] = Field(default_factory=list)

    # Character & Fitness
    character_fitness_statement: Optional[str] = None
    character_fitness_sources: List[str] = Field(default_factory=list)

    # Application fee
    application_fee_statement: Optional[str] = None
    application_fee_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_motion_admission() -> str:
    return """
    Extract the specific information the answer provides regarding North Carolina admission on motion (without examination)
    for a New York attorney and UBE acceptance. Focus strictly on what is explicitly stated in the answer and collect any
    URLs cited.

    Required fields:
    1) reciprocity_statement: The exact sentence or concise paraphrase where the answer states whether North Carolina permits
       admission on motion/reciprocity for attorneys licensed in other states (specifically including New York if mentioned).
       If not stated, return null.
    2) reciprocity_sources: All URLs the answer cites that support North Carolina’s admission on motion eligibility/reciprocity.
       Return an array (can be empty).

    3) practice_rule_text: The exact rule text the answer states for NC’s years-of-practice requirement (e.g., “4 years of active
       practice out of the last 6 years”). If not present, return null.
    4) practice_required_years_rule: A normalized short version of the rule if present (e.g., “4 of past 6”). If missing, return null.
    5) practice_categories_mentioned: List any categories the answer explicitly lists as counting as “practice” for NC admission-on-motion
       purposes (e.g., law teaching, government agency work, military service, in-house corporate practice, service in a judicial court of record).
       Return an array (can be empty).
    6) practice_sources: All URLs cited that discuss NC’s years-of-practice rule or define what counts as “practice”. Return an array.

    7) nc_ube_jurisdiction_statement: The exact statement where the answer says NC is a UBE jurisdiction (if present). Else null.
    8) ube_transfer_acceptance_statement: The exact statement where the answer says NC accepts transferred UBE scores (if present). Else null.
    9) ube_portability_condition_text: The exact portability condition text the answer states (e.g., “MEE, MPT, MBE must be taken in the same
       jurisdiction during the same administration”). If not present, return null.
    10) nc_minimum_ube_score: The numeric minimum UBE score the answer states for NC (e.g., “270”). If not provided, return null. Do NOT invent.
    11) ube_sources: All URLs cited that discuss NC UBE jurisdiction or UBE score transfer and minimum score. Return an array.

    12) character_fitness_statement: The exact statement where the answer mentions that the applicant must satisfy NC character and fitness requirements.
        If not present, return null.
    13) character_fitness_sources: All URLs cited that relate to character & fitness requirements. Return an array.

    14) application_fee_statement: The exact statement where the answer says an application fee must be paid for admission on motion to the NC State Bar.
        If not stated, return null. Do not invent amounts.
    15) application_fee_sources: All URLs cited that relate to application fees. Return an array.

    URL extraction rules:
    - Extract only URLs explicitly present in the answer (plain URLs or markdown links).
    - If a URL is missing a protocol, prepend http://.
    - If no URLs are provided, return an empty array for that field.
    """


# --------------------------------------------------------------------------- #
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #
def parse_int(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    digits = re.findall(r"\d+", value)
    if not digits:
        return None
    try:
        return int(digits[0])
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification Subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_reciprocity_requirement(
    evaluator: Evaluator,
    parent_node,
    data: MotionAdmissionExtraction,
) -> None:
    leaf = evaluator.add_leaf(
        id="Reciprocity_Requirement",
        desc="Verify whether North Carolina permits admission on motion/reciprocity for attorneys licensed in New York (i.e., whether NY attorneys are eligible to apply on motion in NC)",
        parent=parent_node,
        critical=True,
    )
    claim = (
        "North Carolina permits admission on motion (without examination) for attorneys admitted in another U.S. "
        "jurisdiction who meet NC's eligibility requirements; therefore, a New York‑licensed attorney is eligible to apply on motion in NC."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.reciprocity_sources,
        additional_instruction="Accept if NC official rules or Board of Law Examiners materials indicate admission on motion (comity) is available to attorneys admitted in other jurisdictions; New York is included as 'another jurisdiction'."
    )


async def verify_years_of_practice_requirement(
    evaluator: Evaluator,
    parent_node,
    data: MotionAdmissionExtraction,
) -> None:
    main = evaluator.add_parallel(
        id="Years_of_Practice_Requirement",
        desc="Verify and apply North Carolina’s active-practice requirement for admission on motion",
        parent=parent_node,
        critical=True,
    )

    # State NC’s years-of-practice rule
    years_rule_leaf = evaluator.add_leaf(
        id="State_Years_Practice_Rule",
        desc="State North Carolina’s years-of-practice rule for admission on motion (4 years of active practice out of the past 6 years)",
        parent=main,
        critical=True,
    )
    claim_years_rule = (
        "For admission on motion in North Carolina (without examination), the rule requires at least 4 years of active practice of law "
        "within the past 6 years."
    )
    await evaluator.verify(
        claim=claim_years_rule,
        node=years_rule_leaf,
        sources=data.practice_sources,
        additional_instruction="Verify the exact time-in-practice rule from NC official sources; allow minor wording variations such as 'four of the preceding six years'."
    )

    # Assess the attorney’s stated 5 years of active practice
    assess_leaf = evaluator.add_leaf(
        id="Assess_Attorney_Practice_Years",
        desc="Determine whether the attorney’s stated 5 years of active practice satisfies North Carolina’s years-of-practice rule",
        parent=main,
        critical=True,
    )
    claim_assess = (
        "Given North Carolina requires 4 years of active practice within the past 6 years, an attorney who has actively practiced law for "
        "the past 5 years satisfies this requirement."
    )
    await evaluator.verify(
        claim=claim_assess,
        node=assess_leaf,
        additional_instruction="This is a straightforward logical check; 5 years in the past 5 years meets a '4 years in the past 6 years' threshold."
    )

    # State definition of "practice" categories
    def_practice_leaf = evaluator.add_leaf(
        id="State_Definition_of_Practice",
        desc="State what counts as “practice” for NC admission-on-motion purposes, including the listed categories (law teaching, government agency work, military service, in-house corporate practice, and service in a judicial court of record)",
        parent=main,
        critical=True,
    )
    claim_def_practice = (
        "For meeting North Carolina’s admission-on-motion 'active practice' requirement, qualifying practice includes: "
        "law teaching at an accredited law school; legal work for a government agency; military legal service (e.g., JAG); "
        "in-house corporate practice; and service in a judicial court of record."
    )
    await evaluator.verify(
        claim=claim_def_practice,
        node=def_practice_leaf,
        sources=data.practice_sources,
        additional_instruction="Confirm that NC’s definition of qualifying 'practice of law' includes the listed categories; allow synonymous phrasing."
    )


async def verify_ube_score_acceptance(
    evaluator: Evaluator,
    parent_node,
    data: MotionAdmissionExtraction,
) -> None:
    main = evaluator.add_sequential(
        id="UBE_Score_Acceptance",
        desc="Determine whether the attorney’s UBE score can be accepted in North Carolina for admission without retaking the exam",
        parent=parent_node,
        critical=True,
    )

    # Confirm NC is a UBE jurisdiction and accepts transferred UBE scores
    ube_confirm_leaf = evaluator.add_leaf(
        id="Confirm_NC_UBE_Jurisdiction_And_Score_Transfer",
        desc="Confirm whether North Carolina is a UBE jurisdiction and whether it accepts transferred UBE scores for the relevant admission pathway",
        parent=main,
        critical=True,
    )
    claim_ube_confirm = (
        "North Carolina is a Uniform Bar Examination (UBE) jurisdiction and accepts transferred UBE scores for admission (e.g., admission by transfer of UBE score)."
    )
    await evaluator.verify(
        claim=claim_ube_confirm,
        node=ube_confirm_leaf,
        sources=data.ube_sources,
        additional_instruction="Check NC Board of Law Examiners materials or official rules to confirm UBE usage in NC and acceptance of transferred UBE scores."
    )

    # State UBE portability condition (MEE, MPT, MBE in same jurisdiction & same administration)
    portability_leaf = evaluator.add_leaf(
        id="State_UBE_Portability_Condition",
        desc="State the constraint that UBE score transfer requires taking all portions (MEE, MPT, MBE) in the same jurisdiction during the same administration",
        parent=main,
        critical=True,
    )
    claim_portability = (
        "Transfer of a UBE score requires that the applicant took all UBE components—the MEE, MPT, and MBE—in the same jurisdiction during the same administration."
    )
    await evaluator.verify(
        claim=claim_portability,
        node=portability_leaf,
        sources=data.ube_sources,
        additional_instruction="Confirm the standard UBE portability condition from NC or NCBE guidance; allow slight wording variations."
    )

    # State NC minimum UBE score (verify the value stated in the answer against sources)
    min_score_leaf = evaluator.add_leaf(
        id="State_NC_Minimum_UBE_Score",
        desc="State North Carolina’s minimum UBE score requirement applicable to the scenario (do not hard-code a value unless provided)",
        parent=main,
        critical=True,
    )
    # Build a claim reflecting the answer's stated minimum; if none, indicate missing explicitly
    if data.nc_minimum_ube_score and data.nc_minimum_ube_score.strip():
        claim_min_score = f"North Carolina’s minimum transferable UBE score is {data.nc_minimum_ube_score.strip()}."
    else:
        claim_min_score = (
            "The answer does not state a numeric minimum UBE score for North Carolina; therefore, the required minimum score is missing."
        )
    await evaluator.verify(
        claim=claim_min_score,
        node=min_score_leaf,
        sources=data.ube_sources,
        additional_instruction="Only pass if the webpage(s) explicitly indicate the numeric minimum and the answer states such a number; if the answer omits the number, judge this item as unsupported."
    )

    # Compare the attorney’s 268 to NC minimum and conclude acceptance
    compare_leaf = evaluator.add_leaf(
        id="Compare_268_To_Minimum_And_Conclude",
        desc="Compare the attorney’s UBE score (268) against North Carolina’s stated minimum and conclude whether the score is accepted",
        parent=main,
        critical=True,
    )
    min_score_int = parse_int(data.nc_minimum_ube_score)
    if min_score_int is not None:
        meets = 268 >= min_score_int
        conclusion = "is accepted" if meets else "is not accepted"
        claim_compare = (
            f"Given North Carolina’s minimum UBE score is {min_score_int}, a UBE score of 268 "
            f"{'meets' if meets else 'does not meet'} that minimum and therefore {conclusion}."
        )
    else:
        # If the minimum is not available/parseable, this node will be skipped by sequential gating if the previous leaf failed.
        claim_compare = (
            "North Carolina’s minimum UBE score could not be determined from the answer; thus a comparison to 268 cannot be concluded."
        )

    await evaluator.verify(
        claim=claim_compare,
        node=compare_leaf,
        additional_instruction="This is a purely logical comparison based on the numeric minimum; ignore other admission requirements."
    )


async def verify_character_and_fitness(
    evaluator: Evaluator,
    parent_node,
    data: MotionAdmissionExtraction,
) -> None:
    leaf = evaluator.add_leaf(
        id="Character_and_Fitness",
        desc="State that the applicant must satisfy North Carolina character and fitness requirements for bar admission",
        parent=parent_node,
        critical=True,
    )
    claim = "North Carolina requires applicants to satisfy character and fitness requirements as part of bar admission."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.character_fitness_sources,
        additional_instruction="Confirm from NC official sources (e.g., Board of Law Examiners) that character and fitness is required."
    )


async def verify_application_fee(
    evaluator: Evaluator,
    parent_node,
    data: MotionAdmissionExtraction,
) -> None:
    leaf = evaluator.add_leaf(
        id="Application_Fee",
        desc="State that an application fee must be paid for admission on motion to the North Carolina State Bar (do not hard-code the fee amount unless provided)",
        parent=parent_node,
        critical=True,
    )
    claim = "An application fee must be paid to apply for admission on motion to the North Carolina State Bar."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.application_fee_sources,
        additional_instruction="Verify from NC official sources that an application fee is required; do not evaluate any specific amount unless explicitly provided."
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
    # Initialize evaluator (root is non-critical by design in framework; we will add critical children per rubric)
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_motion_admission(),
        template_class=MotionAdmissionExtraction,
        extraction_name="motion_admission_extraction",
    )

    # Optionally record scenario context
    evaluator.add_ground_truth({
        "scenario": {
            "jurisdiction_from": "New York",
            "jurisdiction_to": "North Carolina",
            "active_practice_years": 5,
            "ube_score": 268,
        }
    })

    # Build verification tree according to rubric
    # Root-level critical leaves/groups
    await verify_reciprocity_requirement(evaluator, root, extraction)
    await verify_years_of_practice_requirement(evaluator, root, extraction)
    await verify_ube_score_acceptance(evaluator, root, extraction)
    await verify_character_and_fitness(evaluator, root, extraction)
    await verify_application_fee(evaluator, root, extraction)

    # Return structured summary
    return evaluator.get_summary()