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
TASK_ID = "big_ten_universities_engineering"
TASK_DESCRIPTION = """
Identify three universities in the Big Ten Conference that meet all of the following criteria: 
(1) the university's football stadium has a seating capacity of at least 70,000, 
(2) the university has an undergraduate enrollment of at least 40,000 students, and 
(3) the university offers ABET-accredited engineering programs. 
For each university, provide the university name, its football stadium name and capacity, its undergraduate enrollment figure, 
and reference URLs that verify each of these criteria.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    university_name: Optional[str] = None
    stadium_name: Optional[str] = None
    stadium_capacity: Optional[str] = None
    undergrad_enrollment: Optional[str] = None

    # Source URLs per criterion
    big_ten_member_urls: List[str] = Field(default_factory=list)
    stadium_urls: List[str] = Field(default_factory=list)
    enrollment_urls: List[str] = Field(default_factory=list)
    abet_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
Extract up to 5 universities (we will later consider only the first 3) from the answer that the user provided. 
For each university, extract the following fields exactly as stated in the answer:

- university_name: The name of the university.
- stadium_name: The name of the university's primary football stadium.
- stadium_capacity: The stated seating capacity from the answer (as a string, keep formatting like commas).
- undergrad_enrollment: The stated undergraduate enrollment figure from the answer (as a string, keep formatting like commas).

Also extract URLs explicitly provided in the answer that support each criterion:
- big_ten_member_urls: URLs that directly support or confirm the university is a current member of the Big Ten Conference (e.g., Big Ten official page, university athletics page, NCAA/CFB references). Extract only actual URLs.
- stadium_urls: URLs that directly support the football stadium name and its seating capacity.
- enrollment_urls: URLs that directly support the undergraduate enrollment figure.
- abet_urls: URLs that directly support that the university offers ABET-accredited engineering programs (e.g., ABET Accredited Program Search or an official university page explicitly stating ABET accreditation).

Rules:
1) Extract only information explicitly mentioned in the answer. Do not infer or add missing data.
2) If a field is missing, set it to null (for strings) or [] (for URL lists).
3) For URLs, accept plain URLs or URLs embedded in markdown links; extract the actual URL string.
4) Do not de-duplicate or modify text; preserve capitalization and punctuation.

Return a JSON object:
{
  "universities": [
    {
      "university_name": ...,
      "stadium_name": ...,
      "stadium_capacity": ...,
      "undergrad_enrollment": ...,
      "big_ten_member_urls": [...],
      "stadium_urls": [...],
      "enrollment_urls": [...],
      "abet_urls": [...]
    },
    ...
  ]
}
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _urls_present(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


async def _verify_with_urls_or_fail(
    evaluator: Evaluator,
    claim: str,
    node,
    urls: Optional[List[str]],
    additional_instruction: str
) -> None:
    """
    Verify a claim with URLs if present; if no URLs are provided, mark as failed.
    This prevents unsupported claims from being (incorrectly) marked as correct without evidence.
    """
    if _urls_present(urls):
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=additional_instruction
        )
    else:
        node.score = 0.0
        node.status = "failed"


def _first_n_or_pad(items: List[UniversityItem], n: int) -> List[UniversityItem]:
    result = items[:n]
    while len(result) < n:
        result.append(UniversityItem())
    return result


# --------------------------------------------------------------------------- #
# Verification logic for one university                                       #
# --------------------------------------------------------------------------- #
async def verify_one_university(
    evaluator: Evaluator,
    parent_node,
    u: UniversityItem,
    idx: int
) -> None:
    """
    Build and verify the subtree for one university (index idx in [0..2]).
    Structure aligns with rubric while splitting atomic checks into leaf nodes.
    """

    # University group node (non-critical; allows partial credit across universities)
    uni_node = evaluator.add_parallel(
        id=f"university_{idx+1}",
        desc=f"{['First','Second','Third'][idx]} university correctly identified and verified to meet all specified criteria with all required data provided",
        parent=parent_node,
        critical=False
    )

    # ---------------- University Identification ----------------
    ident_node = evaluator.add_parallel(
        id=f"u{idx+1}_university_identification",
        desc="University name is provided and the university is confirmed as a current member of the Big Ten Conference",
        parent=uni_node,
        critical=True  # per rubric, this block is critical
    )

    # Name provided (existence)
    evaluator.add_custom_node(
        result=bool(u.university_name and u.university_name.strip()),
        id=f"u{idx+1}_university_name_provided",
        desc="University name is provided",
        parent=ident_node,
        critical=True
    )

    # Big Ten membership supported by sources
    bigten_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_big_ten_membership_supported",
        desc="University is confirmed as a current member of the Big Ten Conference by cited sources",
        parent=ident_node,
        critical=True
    )
    membership_claim = f"The university '{u.university_name or ''}' is a current member of the Big Ten Conference."
    await _verify_with_urls_or_fail(
        evaluator,
        membership_claim,
        bigten_leaf,
        u.big_ten_member_urls,
        additional_instruction="Verify on the provided webpage(s) that the named university is listed as a current member of the Big Ten Conference. Allow minor name variations; focus on explicit membership."
    )

    # ---------------- Stadium Information ----------------
    stadium_node = evaluator.add_parallel(
        id=f"u{idx+1}_stadium_information",
        desc="Football stadium name is provided and the stated capacity is at least 70,000 seats",
        parent=uni_node,
        critical=True
    )

    # Stadium data provided
    evaluator.add_custom_node(
        result=bool(u.stadium_name and u.stadium_name.strip()) and bool(u.stadium_capacity and u.stadium_capacity.strip()),
        id=f"u{idx+1}_stadium_fields_provided",
        desc="Football stadium name and capacity are provided",
        parent=stadium_node,
        critical=True
    )

    # Capacity threshold supported (>= 70,000)
    stadium_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_stadium_capacity_supported",
        desc="Stadium capacity is supported by sources and is at least 70,000",
        parent=stadium_node,
        critical=True
    )
    cap_claim = (
        f"The football stadium of '{u.university_name or ''}', named '{u.stadium_name or ''}', "
        f"has a seating capacity of {u.stadium_capacity or ''}, and this capacity is at least 70,000."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        cap_claim,
        stadium_leaf,
        u.stadium_urls,
        additional_instruction="Verify the stadium name and official seating capacity from the provided source(s). If multiple capacities are listed, rely on the official/standard capacity. Confirm that it is ≥ 70,000."
    )

    # ---------------- Enrollment Information ----------------
    enroll_node = evaluator.add_parallel(
        id=f"u{idx+1}_enrollment_information",
        desc="Undergraduate enrollment figure is provided and is at least 40,000 students",
        parent=uni_node,
        critical=True
    )

    # Enrollment provided
    evaluator.add_custom_node(
        result=bool(u.undergrad_enrollment and u.undergrad_enrollment.strip()),
        id=f"u{idx+1}_enrollment_provided",
        desc="Undergraduate enrollment figure is provided",
        parent=enroll_node,
        critical=True
    )

    # Enrollment threshold supported (>= 40,000)
    enroll_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_enrollment_supported",
        desc="Undergraduate enrollment is supported by sources and is at least 40,000",
        parent=enroll_node,
        critical=True
    )
    enrollment_claim = (
        f"The undergraduate enrollment at '{u.university_name or ''}' is {u.undergrad_enrollment or ''}, "
        f"and this number is at least 40,000."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        enrollment_claim,
        enroll_leaf,
        u.enrollment_urls,
        additional_instruction="Verify that the provided undergraduate enrollment figure is supported by the source(s). Allow reasonable rounding variations; confirm it is ≥ 40,000."
    )

    # ---------------- ABET-Accredited Engineering Programs ----------------
    abet_node = evaluator.add_parallel(
        id=f"u{idx+1}_engineering_programs",
        desc="University is verified to offer ABET-accredited engineering programs",
        parent=uni_node,
        critical=True
    )

    # Presence of ABET-related URLs
    evaluator.add_custom_node(
        result=_urls_present(u.abet_urls),
        id=f"u{idx+1}_abet_urls_provided",
        desc="ABET accreditation reference URL(s) are provided",
        parent=abet_node,
        critical=True
    )

    # ABET accreditation supported
    abet_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_abet_supported",
        desc="ABET accreditation is supported by cited sources",
        parent=abet_node,
        critical=True
    )
    abet_claim = f"The university '{u.university_name or ''}' offers ABET-accredited engineering program(s)."
    await _verify_with_urls_or_fail(
        evaluator,
        abet_claim,
        abet_leaf,
        u.abet_urls,
        additional_instruction="Verify that the university has ABET-accredited program(s), ideally via the ABET Accredited Program Search or an official university webpage explicitly stating ABET accreditation."
    )

    # ---------------- Reference URLs existence (global for all criteria) ----------------
    # This node ensures the answer provided URLs covering each criterion (membership, stadium, enrollment, ABET)
    evaluator.add_custom_node(
        result=_urls_present(u.big_ten_member_urls) and _urls_present(u.stadium_urls) and _urls_present(u.enrollment_urls) and _urls_present(u.abet_urls),
        id=f"u{idx+1}_reference_urls",
        desc="Valid reference URLs are provided that support verification of all criteria (membership, stadium, enrollment, ABET)",
        parent=uni_node,
        critical=True
    )


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Big Ten universities with stadium/enrollment/ABET constraints task.
    """

    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Universities evaluated independently
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

    # Extract structured university info
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Limit/pad to exactly 3 universities
    universities = _first_n_or_pad(extracted.universities or [], 3)

    # Build and verify subtree for each university
    for idx, u in enumerate(universities):
        await verify_one_university(evaluator, root, u, idx)

    # Return evaluation summary
    return evaluator.get_summary()