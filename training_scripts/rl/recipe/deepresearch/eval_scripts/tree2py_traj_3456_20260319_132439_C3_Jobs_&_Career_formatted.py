import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "il_superintendent_timeline"
TASK_DESCRIPTION = """
An educator who recently completed a bachelor's degree in education wants to plan a comprehensive career path to become a competitive candidate for superintendent positions in Illinois. They plan to exceed the minimum requirements by completing an EdD degree to strengthen their qualifications. Their intended career plan includes: (1) Completing a master's degree in education (pursued concurrently while teaching), (2) Gaining the required teaching experience for Illinois principal endorsement, (3) Serving as a principal to meet Illinois's administrative experience requirement for superintendent endorsement, (4) Completing an EdD degree to enhance their competitiveness, and (5) Meeting all other Illinois superintendent certification requirements. Assuming they pursue their EdD as a full-time student after completing their principal service, what is the minimum number of years from their current position (bachelor's degree completion) until they complete all planned credentials and become eligible to apply for superintendent certification in Illinois? Provide a detailed breakdown showing each career phase, its duration in years, how phases overlap (if applicable), and reference URLs documenting the Illinois certification requirements. Your answer should identify the absolute minimum timeline assuming optimal progression through each phase.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TimelineExtraction(BaseModel):
    # Phase presence and sequencing (from answer text)
    phases_listed: Optional[bool] = None  # True if the answer explicitly lists all main phases
    overlap_explained_once: Optional[bool] = None  # True if the answer explicitly says master's overlaps with teaching
    edd_sequencing_correct: Optional[bool] = None  # True if the answer places EdD after principal/administrative service

    # Durations used by the answer (in years, if stated; else null)
    teaching_years: Optional[float] = None
    masters_years: Optional[float] = None
    masters_concurrent_with_teaching: Optional[bool] = None
    principal_years: Optional[float] = None
    edd_years: Optional[float] = None
    edd_after_principal: Optional[bool] = None
    total_years: Optional[float] = None  # The single minimum total the answer computes

    # Principal endorsement requirements acknowledged in the answer
    principal_masters_req_mentioned: Optional[bool] = None
    principal_teaching_years_required: Optional[float] = None  # Typically 4
    principal_tests_195_196_mentioned: Optional[bool] = None
    principal_admin_academy_2001_mentioned: Optional[bool] = None

    # Superintendent endorsement requirements acknowledged in the answer
    superintendent_masters_req_mentioned: Optional[bool] = None
    superintendent_admin_experience_required_years: Optional[float] = None  # Typically 2
    superintendent_test_225_mentioned: Optional[bool] = None
    superintendent_admin_academy_2000_mentioned: Optional[bool] = None

    # Reference URLs included in the answer
    principal_requirement_urls: List[str] = Field(default_factory=list)
    superintendent_requirement_urls: List[str] = Field(default_factory=list)
    all_requirement_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_timeline_and_requirements() -> str:
    return """
Extract, from the answer, the structured information needed to evaluate the Illinois superintendent eligibility timeline plan.

Return a JSON object with the following fields:

1) Phase presence and sequencing (booleans; true only if the answer explicitly states the item):
- phases_listed: true if the answer identifies the main planned phases: (a) gaining required teaching experience, (b) completing a master's IN PARALLEL with teaching, (c) serving as a principal/administrator to meet superintendent administrative experience, and (d) completing an EdD AFTER principal service, plus (e) meeting any remaining certification steps.
- overlap_explained_once: true if the answer explicitly explains that the master's is pursued concurrently during the teaching phase so it does not add extra time beyond teaching.
- edd_sequencing_correct: true if the answer places the EdD full-time AFTER completing principal/administrative service and does not overlap it.

2) Durations used in years (numbers, can be decimals; if not stated, set null):
- teaching_years
- masters_years
- masters_concurrent_with_teaching (boolean; true if explicitly stated concurrent/overlapped)
- principal_years
- edd_years
- edd_after_principal (boolean)
- total_years (the single minimum total years computed in the answer)

3) Principal endorsement requirements acknowledged (booleans/numbers; true only if explicitly acknowledged):
- principal_masters_req_mentioned (boolean)
- principal_teaching_years_required (number if mentioned; otherwise null)
- principal_tests_195_196_mentioned (boolean)
- principal_admin_academy_2001_mentioned (boolean)

4) Superintendent endorsement requirements acknowledged (booleans/numbers; true only if explicitly acknowledged):
- superintendent_masters_req_mentioned (boolean)
- superintendent_admin_experience_required_years (number if mentioned; otherwise null)
- superintendent_test_225_mentioned (boolean)
- superintendent_admin_academy_2000_mentioned (boolean)

5) Reference URLs that the answer provides to document Illinois principal/superintendent requirements:
- principal_requirement_urls: array of URLs explicitly cited in the answer that document principal endorsement requirements
- superintendent_requirement_urls: array of URLs explicitly cited in the answer that document superintendent endorsement requirements
- all_requirement_urls: array of ALL requirement-related URLs (deduplicate; include any from both of the above lists)

Rules:
- Extract ONLY what appears explicitly in the answer.
- For URLs, extract actual links (including those in markdown). If none are present, return empty arrays.
- For booleans, return true ONLY if the answer clearly states it; otherwise false or null as appropriate.
- For durations, use numeric years if stated (e.g., "2 years", "3.0 years"). If a range is given (e.g., 3–4 years), pick the exact value the answer chooses for its minimum calculation if stated (e.g., 3). If unclear, return null.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _coalesce_urls(data: TimelineExtraction) -> List[str]:
    if data.all_requirement_urls:
        return list(dict.fromkeys(data.all_requirement_urls))  # dedupe while preserving order
    # fallback to union of principal + superintendent arrays
    combined = list(dict.fromkeys((data.principal_requirement_urls or []) + (data.superintendent_requirement_urls or [])))
    return combined


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_phase_breakdown_and_overlap(evaluator: Evaluator, parent_node, data: TimelineExtraction) -> None:
    node = evaluator.add_parallel(
        id="phase_breakdown_and_overlap",
        desc="Provides a detailed phase-by-phase breakdown with durations (in years) and explicitly explains any allowed overlaps and required sequencing from the question.",
        parent=parent_node,
        critical=True,
    )

    # phases_listed
    leaf1 = evaluator.add_leaf(
        id="phases_listed",
        desc="Identifies the main planned phases: teaching experience, master's completion (concurrent with teaching), qualifying administrative/principal experience, and EdD completion after principal service.",
        parent=node,
        critical=True,
    )
    claim1 = (
        "The answer explicitly identifies all main planned phases: "
        "(1) gaining the required teaching experience, "
        "(2) completing a master's degree concurrently while teaching, "
        "(3) serving as a principal/administrator to satisfy superintendent administrative experience, and "
        "(4) completing an EdD full-time after principal/administrative service, "
        "plus acknowledging remaining certification steps."
    )
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        additional_instruction="Judge based only on the answer text. Allow reasonable synonyms (e.g., 'administrator' for 'principal')."
    )

    # overlap_explained_once
    leaf2 = evaluator.add_leaf(
        id="overlap_explained_once",
        desc="Explicitly explains that the master's is pursued concurrently during the teaching phase (so it is not added as extra time beyond the teaching-experience requirement).",
        parent=node,
        critical=True,
    )
    claim2 = (
        "The answer explicitly explains that the master's degree is pursued concurrently during the teaching phase, "
        "so it does not add extra time beyond the teaching-experience requirement."
    )
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        additional_instruction="Focus on explicit wording that the master's overlaps with teaching and adds zero extra years."
    )

    # edd_sequencing
    leaf3 = evaluator.add_leaf(
        id="edd_sequencing",
        desc="Places the EdD as full-time after completing principal/administrative service (no overlap with principal/administrative service, per the question assumption).",
        parent=node,
        critical=True,
    )
    claim3 = (
        "The answer places the EdD as a full-time program AFTER completing the principal/administrative service, "
        "with no overlap between EdD and principal/administrative service."
    )
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        additional_instruction="Confirm the sequencing is strictly after principal service and not overlapped."
    )


async def _verify_requirement_with_urls_or_fail(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    urls: List[str],
    add_ins: str
) -> None:
    # If there are URLs, perform LLM-as-a-Judge verification grounded by URLs
    if urls:
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=parent,
            critical=True,
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=add_ins
        )
    else:
        # No URLs provided -> directly fail this critical requirement leaf
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=f"{desc} (FAILED: no reference URLs provided in the answer to ground this requirement)",
            parent=parent,
            critical=True
        )


async def verify_certification_requirements_and_citations(evaluator: Evaluator, parent_node, data: TimelineExtraction) -> None:
    node = evaluator.add_parallel(
        id="certification_requirements_and_citations",
        desc="Accounts for Illinois principal and superintendent endorsement prerequisites from the constraints and includes reference URL(s) documenting those requirements.",
        parent=parent_node,
        critical=True,
    )

    # Principal endorsement requirements
    principal_node = evaluator.add_parallel(
        id="principal_endorsement_requirements_accounted",
        desc="Accounts for Illinois principal endorsement prerequisites listed in the constraints.",
        parent=node,
        critical=True,
    )

    p_urls = data.principal_requirement_urls or []

    await _verify_requirement_with_urls_or_fail(
        evaluator,
        principal_node,
        "principal_masters_requirement",
        "Accounts for the requirement of a master's degree or higher from a regionally accredited institution for principal endorsement.",
        "Illinois principal endorsement requires a master's degree or higher from a regionally accredited institution.",
        p_urls,
        add_ins="Verify that at least one provided URL explicitly states that a master's (or higher) degree is required for the Illinois principal endorsement. Prefer official ISBE or equivalent authoritative sources."
    )

    await _verify_requirement_with_urls_or_fail(
        evaluator,
        principal_node,
        "principal_teaching_experience_requirement",
        "Accounts for the required 4 years of teaching/school support personnel experience for principal endorsement.",
        "Illinois principal endorsement requires 4 years of full-time teaching or school support personnel experience.",
        p_urls,
        add_ins="Confirm the URL(s) explicitly indicate 4 years of full-time teaching or school support personnel experience is required for the Illinois principal endorsement."
    )

    await _verify_requirement_with_urls_or_fail(
        evaluator,
        principal_node,
        "principal_tests_requirement",
        "Accounts for passing the Principal as an Instructional Leader tests (195 & 196) (time can be assumed minimal but must be acknowledged).",
        "Illinois principal endorsement requires passing the Principal as an Instructional Leader tests, test codes 195 and 196 (or their current equivalents).",
        p_urls,
        add_ins="Look for explicit mention of the Principal as an Instructional Leader content tests 195 and 196 (or clearly equivalent current codes) as a requirement."
    )

    await _verify_requirement_with_urls_or_fail(
        evaluator,
        principal_node,
        "principal_admin_academy_requirement",
        "Accounts for completion of Administrator Academy 2001 (time can be assumed minimal but must be acknowledged).",
        "For Illinois principal endorsement, completion of Administrator Academy 2001 is required.",
        p_urls,
        add_ins="Verify that Administrator Academy 2001 is a stated requirement for Illinois principal endorsement in at least one cited URL."
    )

    # Superintendent endorsement requirements
    super_node = evaluator.add_parallel(
        id="superintendent_endorsement_requirements_accounted",
        desc="Accounts for Illinois superintendent endorsement prerequisites listed in the constraints.",
        parent=node,
        critical=True,
    )

    s_urls = data.superintendent_requirement_urls or []

    await _verify_requirement_with_urls_or_fail(
        evaluator,
        super_node,
        "superintendent_masters_requirement",
        "Accounts for the requirement of a master's degree or higher from a regionally accredited institution for superintendent endorsement.",
        "Illinois superintendent endorsement requires a master's degree or higher from a regionally accredited institution.",
        s_urls,
        add_ins="Confirm that at least one cited URL states a master's (or higher) is required for the Illinois superintendent endorsement. Prefer official ISBE or equivalent authoritative sources."
    )

    await _verify_requirement_with_urls_or_fail(
        evaluator,
        super_node,
        "superintendent_admin_experience_requirement",
        "Accounts for the required 2 years full-time qualifying administrative experience (e.g., principal) while holding a valid administrator license.",
        "Illinois superintendent endorsement requires 2 years of full-time qualifying administrative experience (such as serving as a principal) while holding a valid administrator/educator license.",
        s_urls,
        add_ins="Look for explicit mention of the 2-year full-time administrative (e.g., principal) experience requirement, and that it must be while holding a valid license."
    )

    await _verify_requirement_with_urls_or_fail(
        evaluator,
        super_node,
        "superintendent_test_requirement",
        "Accounts for passing the Superintendent test (225) (time can be assumed minimal but must be acknowledged).",
        "Illinois superintendent endorsement requires passing the Superintendent test, code 225 (or its current equivalent).",
        s_urls,
        add_ins="Verify explicit mention of the Superintendent test (code 225 or clearly current equivalent) as a requirement."
    )

    await _verify_requirement_with_urls_or_fail(
        evaluator,
        super_node,
        "superintendent_admin_academy_requirement",
        "Accounts for completion of Administrator Academy 2000 (time can be assumed minimal but must be acknowledged).",
        "For Illinois superintendent endorsement, completion of Administrator Academy 2000 is required.",
        s_urls,
        add_ins="Verify that Administrator Academy 2000 is a stated requirement for Illinois superintendent endorsement in at least one cited URL."
    )

    # At least one requirements reference URL present (existence check)
    urls_any = _coalesce_urls(data)
    evaluator.add_custom_node(
        result=bool(urls_any),
        id="requirements_reference_urls",
        desc="Provides at least one reference URL documenting Illinois principal and superintendent endorsement requirements relevant to the constraints above.",
        parent=node,
        critical=True
    )


async def verify_duration_assumptions_match_constraints(evaluator: Evaluator, parent_node, data: TimelineExtraction) -> None:
    node = evaluator.add_parallel(
        id="duration_assumptions_match_constraints",
        desc="Uses durations consistent with the provided constraints and states assumptions needed for an 'absolute minimum' timeline.",
        parent=parent_node,
        critical=True,
    )

    # Master's ~2 years
    leaf1 = evaluator.add_leaf(
        id="masters_duration_used",
        desc="Uses a master's duration consistent with the constraint (typically ~2 years).",
        parent=node,
        critical=True
    )
    claim1 = (
        "The answer uses a master's program duration of approximately two years (about 2.0 years) for planning purposes, "
        "even though it overlaps with teaching and thus does not add extra time beyond the teaching-experience phase."
    )
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        additional_instruction="Accept phrasing like 'about 2 years' or 'typically two years'; small rounding is fine."
    )

    # EdD 3–4 years, choose minimum (3) for absolute-minimum timeline
    leaf2 = evaluator.add_leaf(
        id="edd_duration_used",
        desc="Uses an EdD full-time duration consistent with the constraint range (3–4 years) and, for an absolute-minimum timeline, selects the minimum feasible value within that range and states it explicitly.",
        parent=node,
        critical=True
    )
    claim2 = (
        "The answer uses an EdD full-time duration within the 3–4 year range and explicitly selects the minimum feasible value "
        "(e.g., 3 years) for the absolute-minimum timeline calculation."
    )
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        additional_instruction="Confirm the answer selects the minimum feasible EdD duration (ideally 3 years) and states this clearly."
    )


async def verify_total_minimum_timeline(evaluator: Evaluator, parent_node, data: TimelineExtraction) -> None:
    node = evaluator.add_parallel(
        id="total_minimum_timeline_correct_and_consistent",
        desc="Computes a single minimum total time (in years) that is internally consistent with the phase durations, overlap rules, and required sequencing described in the answer.",
        parent=parent_node,
        critical=True,
    )

    # Internal-consistency check for the computed total (expecting 9 years if master's ~2 years overlaps teaching 4 yrs, plus 2 yrs principal, plus 3 yrs EdD)
    leaf1 = evaluator.add_leaf(
        id="total_calculation_consistency",
        desc="Total years are computed consistently from the described phases and any overlaps (i.e., no double-counting overlapped time).",
        parent=node,
        critical=True
    )
    claim1 = (
        "The answer's single minimum total is computed consistently with: "
        "4 years of teaching (with the master's overlapping and thus not adding extra time), "
        "then 2 years of principal/administrative experience, "
        "then a 3-year full-time EdD afterwards, "
        "for a total of 9 years with no double-counting or gaps."
    )
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        additional_instruction="Check that the answer’s total equals 9 years and is consistent with the stated overlap and sequencing."
    )

    # Minimum-claim support explanation
    leaf2 = evaluator.add_leaf(
        id="minimum_claim_supported",
        desc="Explains why the schedule is the absolute minimum under the stated constraints/assumptions (e.g., no gaps, overlaps used where allowed, minimum durations chosen within given ranges).",
        parent=node,
        critical=True
    )
    claim2 = (
        "The answer explicitly explains why its schedule is the absolute minimum under the assumptions: "
        "it uses overlap where allowed (master's concurrent with teaching), selects minimum feasible durations "
        "(e.g., 3 years for EdD), avoids gaps, and follows the required sequencing."
    )
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        additional_instruction="Look for an explicit justification that the proposed schedule is the absolute minimum."
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
    Evaluate an answer for the Illinois superintendent minimum timeline planning task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation for overall judgment
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
    timeline_data = await evaluator.extract(
        prompt=prompt_extract_timeline_and_requirements(),
        template_class=TimelineExtraction,
        extraction_name="timeline_and_requirements_extraction"
    )

    # Build and verify subtrees
    await verify_phase_breakdown_and_overlap(evaluator, root, timeline_data)
    await verify_certification_requirements_and_citations(evaluator, root, timeline_data)
    await verify_duration_assumptions_match_constraints(evaluator, root, timeline_data)
    await verify_total_minimum_timeline(evaluator, root, timeline_data)

    # Return standardized summary
    return evaluator.get_summary()