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
TASK_ID = "mn_superintendent_mar2026"
TASK_DESCRIPTION = (
    "As of March 2026, identify the specific superintendent position at Minnesota's largest public school district "
    "that meets all of the following criteria: (1) the district must serve approximately 38,000 students, "
    "(2) the current superintendent Cory McIntyre's contract must end on June 30, 2026, "
    "(3) the School Board must be determining a leadership plan before the 2026-2027 school year, "
    "(4) candidates must hold at least a master's degree in educational leadership or a related field, "
    "(5) candidates must be eligible for or hold Minnesota superintendent licensure which requires completion of at least 320 hours of field experience, "
    "and (6) candidates must have proven administrative experience at educational institutions. "
    "Provide the district name, current superintendent's name, contract end date, approximate student enrollment, "
    "and the URL reference confirming these details."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SuperintendentPositionExtraction(BaseModel):
    # Core identification fields extracted from the answer
    district_name: Optional[str] = None
    state: Optional[str] = None
    position_title: Optional[str] = None

    # Output details
    student_enrollment_text: Optional[str] = None
    superintendent_name: Optional[str] = None
    contract_end_date_text: Optional[str] = None

    # Policy/timing details
    leadership_plan_timing_text: Optional[str] = None

    # Candidate requirement details
    degree_requirement_text: Optional[str] = None
    licensure_requirement_text: Optional[str] = None
    field_experience_requirement_text: Optional[str] = None
    administrative_experience_requirement_text: Optional[str] = None

    # Optional board-valued experience mentions
    values_budget_reduction_experience_text: Optional[str] = None
    values_literacy_read_act_experience_text: Optional[str] = None

    # Timeframe framing in the answer
    timeframe_statement: Optional[str] = None

    # All URLs explicitly present in the answer that substantiate the above items
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_position_info() -> str:
    return """
    Extract, verbatim from the answer, the following fields about the identified superintendent position and its district. 
    Do NOT infer or rewrite; copy exact text if present. If a field is missing in the answer, return null (or an empty list for URLs).
    Fields to extract:
    - district_name: the exact name of the district identified in the answer (e.g., "Anoka-Hennepin School District", "Anoka-Hennepin Schools", "ISD 11").
    - state: the state where the district is located (e.g., "Minnesota", "MN").
    - position_title: the exact position title identified (e.g., "Superintendent", "Superintendent of Schools").
    - student_enrollment_text: the approximate student enrollment as referenced in the answer (e.g., "about 38,000", "approximately 38,000").
    - superintendent_name: the current superintendent's name.
    - contract_end_date_text: the superintendent contract end date text (e.g., "June 30, 2026", "through June 30, 2026").
    - leadership_plan_timing_text: the text in the answer indicating the School Board is determining a leadership plan before the 2026–2027 school year.
    - degree_requirement_text: the text specifying a minimum of a master's degree in educational leadership or a related field.
    - licensure_requirement_text: the text specifying eligibility for or holding Minnesota superintendent licensure.
    - field_experience_requirement_text: the text specifying at least 320 hours of field experience for MN superintendent licensure.
    - administrative_experience_requirement_text: the text specifying proven administrative experience at educational institutions.
    - values_budget_reduction_experience_text: the text indicating the School Board values experience leading significant budget reductions (if mentioned).
    - values_literacy_read_act_experience_text: the text indicating the School Board values experience advancing literacy changes, including READ Act implementation (if mentioned).
    - timeframe_statement: any explicit phrasing indicating "as of March 2026" (accept reasonable variants like "As of Mar. 2026", "as of 3/2026").
    - sources: list all URLs in the answer that substantiate the claims above. Include every URL explicitly written in the answer; preserve full URLs.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_superintendent_position(
    evaluator: Evaluator,
    parent_node,
    extracted: SuperintendentPositionExtraction,
) -> None:
    """
    Build the verification tree under a task-level parallel node and run verifications.
    Critical nodes correspond to required criteria in the task/rubric.
    """
    # Create a main parallel node for this task (non-critical to allow partial credit display;
    # Criticality is handled at the leaf level as per rubric items)
    main_node = evaluator.add_parallel(
        id="MN_Largest_District_Superintendent_Position_AsOf_Mar2026",
        desc="Evaluate whether the response identifies the superintendent position matching the question/constraints (as of March 2026) and provides the required outputs with appropriate URL evidence.",
        parent=parent_node,
        critical=False
    )

    # Prepare sources from the answer
    sources_list = extracted.sources if extracted and extracted.sources else []
    district_display = extracted.district_name or "the identified district"

    # 1) Critical presence of URL references (as a gate)
    urls_present = bool(sources_list)
    evaluator.add_custom_node(
        result=urls_present,
        id="URL_References_Provided_For_Required_Details",
        desc="Response provides URL reference(s) that substantiate the required output details (district name, superintendent name, contract end date, approximate student enrollment).",
        parent=main_node,
        critical=True
    )

    # 2) Timeframe explicitly addressed in the answer (simple verification against the answer text)
    timeframe_node = evaluator.add_leaf(
        id="Timeframe_AsOf_March_2026_Addressed",
        desc="Response explicitly frames the identification as valid as of March 2026.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly frames the identification as valid as of March 2026.",
        node=timeframe_node,
        additional_instruction="Judge solely from the answer text. Accept clear variants like 'As of Mar. 2026', 'as of 3/2026', or an equivalent explicit timeframe statement indicating March 2026."
    )

    # 3) Build remaining verifications (most require URL grounding)
    claims_and_sources: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # 3.1 Identified position is a superintendent role at a public school district
    node_pos_public = evaluator.add_leaf(
        id="Position_Is_Public_District_Superintendent",
        desc="Identified position is a superintendent role at a public school district.",
        parent=main_node,
        critical=True
    )
    claim_pos_public = f"The position is the superintendent of {district_display}, which is a public K–12 school district (not a private or charter organization)."
    claims_and_sources.append((
        claim_pos_public,
        sources_list,
        node_pos_public,
        "Use the provided URLs (official district, board, or posting pages) to confirm the role is 'Superintendent' and the organization is a public school district. Accept synonyms like 'Superintendent of Schools'."
    ))

    # 3.2 District is in Minnesota
    node_in_mn = evaluator.add_leaf(
        id="District_Is_In_Minnesota",
        desc="Identified public school district is located in Minnesota.",
        parent=main_node,
        critical=True
    )
    claim_in_mn = f"{district_display} is located in Minnesota (MN)."
    claims_and_sources.append((
        claim_in_mn,
        sources_list,
        node_in_mn,
        "Confirm from the source that the district is in Minnesota. Accept abbreviations like 'MN' and identifiers like 'ISD 11' that imply Minnesota."
    ))

    # 3.3 District identified is Anoka-Hennepin (name match against the answer text)
    node_is_ah = evaluator.add_leaf(
        id="District_Is_Anoka_Hennepin",
        desc="Identified district is Anoka-Hennepin School District.",
        parent=main_node,
        critical=True
    )
    claim_is_ah = (
        f"The district identified in the answer refers to Anoka-Hennepin School District "
        f"(also known as 'Anoka-Hennepin Schools' or 'Independent School District 11' / 'ISD 11'). "
        f"Extracted district name (if any): '{extracted.district_name or ''}'."
    )
    claims_and_sources.append((
        claim_is_ah,
        None,  # Judge solely from the answer text (name matching)
        node_is_ah,
        "Judge based only on the answer text; allow reasonable variants like 'Anoka-Hennepin Schools', 'Anoka-Hennepin ISD 11', or similar."
    ))

    # 3.4 Largest district in Minnesota by enrollment
    node_largest = evaluator.add_leaf(
        id="District_Is_Largest_By_Enrollment_In_MN",
        desc="Identified district is Minnesota’s largest public school district by student enrollment.",
        parent=main_node,
        critical=True
    )
    claim_largest = f"{district_display} is Minnesota’s largest public school district by student enrollment."
    claims_and_sources.append((
        claim_largest,
        sources_list,
        node_largest,
        "Look for wording like 'state's largest school district' or similar on official pages or reputable sources."
    ))

    # 3.5 Serves approximately 38,000 students
    node_enroll = evaluator.add_leaf(
        id="District_Serves_Approximately_38000_Students",
        desc="Identified district serves approximately 38,000 students.",
        parent=main_node,
        critical=True
    )
    claim_enroll = "The district serves approximately 38,000 students."
    claims_and_sources.append((
        claim_enroll,
        sources_list,
        node_enroll,
        "Accept approximate phrasings and minor rounding (e.g., 37k–39.5k)."
    ))

    # 3.6 Current superintendent is Cory McIntyre
    node_super = evaluator.add_leaf(
        id="Current_Superintendent_Is_Cory_McIntyre",
        desc="Response identifies the current superintendent as Cory McIntyre.",
        parent=main_node,
        critical=True
    )
    claim_super = "The current superintendent is Cory McIntyre."
    claims_and_sources.append((
        claim_super,
        sources_list,
        node_super,
        "Confirm from official district/board pages or recent announcements."
    ))

    # 3.7 Contract ends June 30, 2026
    node_contract = evaluator.add_leaf(
        id="Superintendent_Contract_Ends_June_30_2026",
        desc="Response states the current superintendent’s contract ends on June 30, 2026.",
        parent=main_node,
        critical=True
    )
    claim_contract = "Cory McIntyre's superintendent contract ends on June 30, 2026."
    claims_and_sources.append((
        claim_contract,
        sources_list,
        node_contract,
        "Accept equivalent wording like 'through June 30, 2026' or 'ending 6/30/2026'."
    ))

    # 3.8 Board determining leadership plan before 2026–2027
    node_plan = evaluator.add_leaf(
        id="Board_Determining_Leadership_Plan_Before_2026_2027",
        desc="Response states the School Board is determining a leadership plan before the 2026–2027 school year.",
        parent=main_node,
        critical=True
    )
    claim_plan = "The School Board is determining a leadership plan before the 2026–2027 school year."
    claims_and_sources.append((
        claim_plan,
        sources_list,
        node_plan,
        "Accept variants like 'ahead of the 2026-27 school year' or 'before SY 2026-27'."
    ))

    # 3.9 Degree requirement (master's in educational leadership or related)
    node_degree = evaluator.add_leaf(
        id="Candidate_Degree_Requirement_Masters",
        desc="Response states candidates must hold at least a master’s degree in educational leadership or a related field.",
        parent=main_node,
        critical=True
    )
    claim_degree = "Candidates must hold at least a master's degree in educational leadership or a closely related field."
    claims_and_sources.append((
        claim_degree,
        sources_list,
        node_degree,
        "Look for job posting or official qualification pages indicating a minimum of a master's degree (educational leadership/administration or related)."
    ))

    # 3.10 Licensure requirement (MN superintendent)
    node_license = evaluator.add_leaf(
        id="Candidate_Licensure_Requirement_MN_Superintendent",
        desc="Response states candidates must be eligible for or hold Minnesota superintendent licensure.",
        parent=main_node,
        critical=True
    )
    claim_license = "Candidates must be eligible for or hold Minnesota superintendent licensure."
    claims_and_sources.append((
        claim_license,
        sources_list,
        node_license,
        "Accept phrases like 'must qualify for a Minnesota superintendent license' (MN Board of School Administrators/MDE)."
    ))

    # 3.11 MN licensure requires at least 320 hours of field experience
    node_320 = evaluator.add_leaf(
        id="MN_Licensure_Requires_320_Hours_Field_Experience",
        desc="Response states Minnesota superintendent licensure requires completion of at least 320 hours of field experience.",
        parent=main_node,
        critical=True
    )
    claim_320 = "Minnesota superintendent licensure requires completion of at least 320 hours of field experience."
    claims_and_sources.append((
        claim_320,
        sources_list,
        node_320,
        "Accept clear equivalents like 'minimum of 320 hours', '≥ 320 hours', or 'superintendent internship of at least 320 hours'."
    ))

    # 3.12 Administrative experience required
    node_admin = evaluator.add_leaf(
        id="Candidate_Administrative_Experience_Required",
        desc="Response states candidates have proven administrative experience at educational institutions.",
        parent=main_node,
        critical=True
    )
    claim_admin = "Candidates must have proven administrative experience at educational institutions."
    claims_and_sources.append((
        claim_admin,
        sources_list,
        node_admin,
        "Accept synonyms like 'successful administrative leadership experience' at school or district level."
    ))

    # 3.13 Non-critical: Board values budget reduction experience
    node_budget = evaluator.add_leaf(
        id="Board_Values_Budget_Reduction_Experience",
        desc="Response mentions that the School Board values experience leading significant budget reductions.",
        parent=main_node,
        critical=False
    )
    claim_budget = "The School Board values experience leading significant budget reductions."
    claims_and_sources.append((
        claim_budget,
        sources_list,
        node_budget,
        "This is optional context. Verify only if the provided sources mention prioritizing or valuing budget reduction leadership experience."
    ))

    # 3.14 Non-critical: Board values literacy/READ Act experience
    node_literacy = evaluator.add_leaf(
        id="Board_Values_Literacy_READ_Act_Experience",
        desc="Response mentions that the School Board values experience advancing literacy changes including READ Act implementation.",
        parent=main_node,
        critical=False
    )
    claim_literacy = "The School Board values experience advancing literacy changes, including READ Act implementation."
    claims_and_sources.append((
        claim_literacy,
        sources_list,
        node_literacy,
        "This is optional context. Verify only if the provided sources mention valuing literacy initiatives or READ Act implementation experience."
    ))

    # Run all URL-grounded checks in parallel (timeframe already checked; URL presence already added)
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the Minnesota largest district superintendent position (as of March 2026).
    """
    # Initialize evaluator (root is non-critical, parallel by default)
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
        default_model=model
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_position_info(),
        template_class=SuperintendentPositionExtraction,
        extraction_name="position_extraction"
    )

    # Optional: record expected reference info for transparency (not used for scoring)
    evaluator.add_ground_truth({
        "expected_district": "Anoka-Hennepin School District (Anoka-Hennepin Schools, ISD 11)",
        "expected_superintendent": "Cory McIntyre",
        "expected_contract_end_date": "June 30, 2026",
        "expected_enrollment_approx": "approximately 38,000 students",
        "timeframe": "As of March 2026"
    })

    # Build tree and run verifications
    await verify_superintendent_position(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()