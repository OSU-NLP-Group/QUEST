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
TASK_ID = "us_state_ag_criteria_2024_2026"
TASK_DESCRIPTION = (
    "Identify the state Attorney General in the United States who meets all of the following criteria: "
    "(1) Assumed office as Attorney General between December 1, 2024 and February 26, 2026; "
    "(2) Previously served as a member of their state legislature; "
    "(3) Served as Speaker of their state House of Representatives; "
    "(4) Holds a Juris Doctor degree from a law school located in the same state where they currently serve as Attorney General; "
    "(5) Was admitted to their state bar in 2006; "
    "(6) Earned their undergraduate degree from a university located in the same state where they currently serve as Attorney General; "
    "(7) Spent at least 15 years in private legal practice before becoming Attorney General. "
    "Provide the following information about this attorney general: "
    "(a) The name of the law school where they earned their Juris Doctor; "
    "(b) The name of the university where they earned their undergraduate degree."
)

DATE_RANGE_START = "December 1, 2024"
DATE_RANGE_END = "February 26, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AGInfo(BaseModel):
    # Core identity
    name: Optional[str] = None
    state: Optional[str] = None  # Full state name preferred (e.g., "Nebraska")

    # Office / chronology
    assumed_office_date: Optional[str] = None  # e.g., "January 13, 2025"

    # Legislature background
    previously_state_legislator: Optional[str] = None  # "yes"/"no"/details as string
    served_as_state_house_speaker: Optional[str] = None  # "yes"/"no"/details as string

    # Education
    law_school_name: Optional[str] = None
    law_school_state: Optional[str] = None
    undergrad_university_name: Optional[str] = None
    undergrad_university_state: Optional[str] = None

    # Bar admission
    bar_admission_state: Optional[str] = None
    bar_admission_year: Optional[str] = None  # Keep as string for flexibility

    # Practice background
    private_practice_summary: Optional[str] = None
    private_practice_duration_years: Optional[str] = None  # Keep flexible (e.g., "15+" or "approx. 16")

    # Sources (URLs)
    sources_general: List[str] = Field(default_factory=list)
    sources_office: List[str] = Field(default_factory=list)
    sources_legislature: List[str] = Field(default_factory=list)
    sources_speaker: List[str] = Field(default_factory=list)
    sources_jd: List[str] = Field(default_factory=list)
    sources_bar_admission: List[str] = Field(default_factory=list)
    sources_undergrad: List[str] = Field(default_factory=list)
    sources_private_practice: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ag_info() -> str:
    return """
    Extract information about a single identified current U.S. state Attorney General (AG) that the answer claims meets all the listed criteria.
    If multiple individuals are mentioned, select the one the answer asserts satisfies ALL criteria; otherwise select the first clearly identified AG.

    Return a JSON object with the following fields (use null when not present in the answer):

    Identity:
    - name: The full name of the identified Attorney General.
    - state: The full name of the U.S. state where they currently serve as Attorney General (e.g., "Arkansas", not abbreviation).

    Office / chronology:
    - assumed_office_date: The specific date (as written in the answer) when they assumed office as Attorney General.

    Legislature background:
    - previously_state_legislator: A brief string indicating whether they previously served in their state legislature (e.g., "yes", or "Member of the State House").
    - served_as_state_house_speaker: A brief string indicating whether they served as Speaker of the state House of Representatives (e.g., "yes", or "Speaker of the House, 2019–2021").

    Education:
    - law_school_name: The name of the law school where they earned their Juris Doctor (JD).
    - law_school_state: The state where that law school is located.
    - undergrad_university_name: The name of the university where they earned their undergraduate degree.
    - undergrad_university_state: The state where that undergraduate university is located.

    Bar admission:
    - bar_admission_state: The state bar they were admitted to (ideally the same state as their current AG role).
    - bar_admission_year: The year they were admitted to the state bar (e.g., "2006").

    Private practice background:
    - private_practice_summary: A short summary of their private legal practice (e.g., firm names, years).
    - private_practice_duration_years: The number of years in private practice if explicitly stated (keep as a string; examples: "15", "16", "15+", or "approx. 20").

    Sources (URLs explicitly present in the answer; include only valid URLs):
    - sources_general: All general URLs about the person or their official bio pages.
    - sources_office: URLs supporting AG status and/or assumed office date.
    - sources_legislature: URLs supporting prior service in the state legislature.
    - sources_speaker: URLs supporting service as Speaker of the state House of Representatives (or equivalent).
    - sources_jd: URLs supporting the JD degree and the law school's location/state.
    - sources_bar_admission: URLs supporting bar admission and year.
    - sources_undergrad: URLs supporting the undergraduate degree and the university's location/state.
    - sources_private_practice: URLs supporting private practice career duration/details.

    Rules:
    - Extract only information explicitly mentioned in the answer; do not infer or add facts.
    - For URL fields, extract only actual URLs explicitly present in the answer (plain links or markdown links).
    - If a URL lacks a protocol, prepend "http://".
    - If any field or URL list is not mentioned, set it to null or an empty list accordingly.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*source_lists: Optional[List[str]]) -> List[str]:
    """Merge multiple lists of URLs, preserve order, deduplicate, and ignore empty/invalid strings."""
    seen = set()
    merged: List[str] = []
    for lst in source_lists:
        if not lst:
            continue
        for url in lst:
            if not isinstance(url, str):
                continue
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _pick_sources(ag: AGInfo, primary: Optional[List[str]], *fallbacks: Optional[List[str]]) -> Optional[List[str]]:
    """
    Prefer primary sources; if none, use fallbacks; return None if nothing available (to trigger simple verification).
    """
    merged = _merge_sources(primary, *fallbacks)
    return merged if merged else None


def _nz(s: Optional[str]) -> str:
    """Normalize None/empty to a placeholder for claims."""
    return s.strip() if isinstance(s, str) and s.strip() else "UNKNOWN"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree_and_run(evaluator: Evaluator, ag: AGInfo) -> None:
    """
    Build the verification tree (following the rubric) and run all checks.
    """
    # Main task node (critical, sequential) under root
    main_task = evaluator.add_sequential(
        id="Complete_Research_Task",
        desc="Identify the US state Attorney General who meets all specified criteria and provide the required educational institution names.",
        parent=evaluator.root,
        critical=True
    )

    # Identify & Verify AG (critical, sequential)
    identify_node = evaluator.add_sequential(
        id="Identify_And_Verify_Attorney_General",
        desc="Name an Attorney General and verify they satisfy all listed constraints.",
        parent=main_task,
        critical=True
    )

    # 1) Provide_AG_Name (critical leaf)
    provide_name_leaf = evaluator.add_leaf(
        id="Provide_AG_Name",
        desc="Provide the individual's name (sufficient to identify the Attorney General).",
        parent=identify_node,
        critical=True
    )
    name_claim = f"The answer provides the individual's name: '{_nz(ag.name)}'."
    await evaluator.verify(
        claim=name_claim,
        node=provide_name_leaf,
        sources=None,
        additional_instruction="This check only verifies that a non-empty name is presented in the answer. If the extracted name is UNKNOWN or empty, mark incorrect."
    )

    # 2) Meets_All_Criteria (critical, parallel)
    criteria_node = evaluator.add_parallel(
        id="Meets_All_Criteria",
        desc="Verify the identified individual meets every constraint in the prompt.",
        parent=identify_node,
        critical=True
    )

    # 2.a) Is_Current_US_State_AG
    is_ag_leaf = evaluator.add_leaf(
        id="Is_Current_US_State_AG",
        desc="The individual is a current state Attorney General in the United States.",
        parent=criteria_node,
        critical=True
    )
    is_ag_claim = f"{_nz(ag.name)} is the Attorney General of {_nz(ag.state)}."
    await evaluator.verify(
        claim=is_ag_claim,
        node=is_ag_leaf,
        sources=_pick_sources(ag, ag.sources_office, ag.sources_general),
        additional_instruction="Use the provided URLs (official state AG site, bio pages, or reliable sources) to confirm the person currently holds the office of Attorney General in the stated state."
    )

    # 2.b) Assumed_Office_In_Range
    office_range_leaf = evaluator.add_leaf(
        id="Assumed_Office_In_Range",
        desc=f"Assumed office as Attorney General between {DATE_RANGE_START} and {DATE_RANGE_END}.",
        parent=criteria_node,
        critical=True
    )
    office_range_claim = (
        f"{_nz(ag.name)} assumed office as Attorney General on '{_nz(ag.assumed_office_date)}', "
        f"which falls between {DATE_RANGE_START} and {DATE_RANGE_END}."
    )
    await evaluator.verify(
        claim=office_range_claim,
        node=office_range_leaf,
        sources=_pick_sources(ag, ag.sources_office, ag.sources_general),
        additional_instruction=(
            f"Confirm the assumed office date from the source and judge whether it is within the inclusive range "
            f"{DATE_RANGE_START} to {DATE_RANGE_END}."
        )
    )

    # 2.c) Previously_State_Legislator
    legislator_leaf = evaluator.add_leaf(
        id="Previously_State_Legislator",
        desc="Previously served as a member of their state legislature.",
        parent=criteria_node,
        critical=True
    )
    legislator_claim = (
        f"{_nz(ag.name)} previously served as a member of the {_nz(ag.state)} state legislature."
    )
    await evaluator.verify(
        claim=legislator_claim,
        node=legislator_leaf,
        sources=_pick_sources(ag, ag.sources_legislature, ag.sources_general),
        additional_instruction="Accept prior service in either the state House or state Senate as 'member of the state legislature', as supported by the cited webpages."
    )

    # 2.d) Served_As_State_House_Speaker
    speaker_leaf = evaluator.add_leaf(
        id="Served_As_State_House_Speaker",
        desc="Served as Speaker of their state House of Representatives.",
        parent=criteria_node,
        critical=True
    )
    speaker_claim = (
        f"{_nz(ag.name)} served as Speaker of the {_nz(ag.state)} House of Representatives."
    )
    await evaluator.verify(
        claim=speaker_claim,
        node=speaker_leaf,
        sources=_pick_sources(ag, ag.sources_speaker, ag.sources_legislature, ag.sources_general),
        additional_instruction=(
            "Verify service as 'Speaker of the House'. If the state's legislature uses a different name but is functionally equivalent "
            "(e.g., unicameral 'Speaker of the Legislature'), treat it as satisfying the criterion if sources clearly support it."
        )
    )

    # 2.e) JD_From_In_State_Law_School
    jd_leaf = evaluator.add_leaf(
        id="JD_From_In_State_Law_School",
        desc="Holds a Juris Doctor degree from a law school located in the same state where they currently serve as Attorney General.",
        parent=criteria_node,
        critical=True
    )
    jd_claim = (
        f"{_nz(ag.name)} holds a Juris Doctor from '{_nz(ag.law_school_name)}', located in {_nz(ag.law_school_state)}, "
        f"the same state where they currently serve as Attorney General ({_nz(ag.state)})."
    )
    await evaluator.verify(
        claim=jd_claim,
        node=jd_leaf,
        sources=_pick_sources(ag, ag.sources_jd, ag.sources_general),
        additional_instruction=(
            "Confirm both (1) that the person has a JD from the named institution and (2) that the institution is located in the same state as the AG's current state. "
            "Use official bios, school pages, or other reliable sources."
        )
    )

    # 2.f) Admitted_To_State_Bar_2006
    bar_leaf = evaluator.add_leaf(
        id="Admitted_To_State_Bar_2006",
        desc="Was admitted to their state bar in 2006.",
        parent=criteria_node,
        critical=True
    )
    bar_claim = (
        f"{_nz(ag.name)} was admitted to the {_nz(ag.state)} state bar in 2006."
    )
    await evaluator.verify(
        claim=bar_claim,
        node=bar_leaf,
        sources=_pick_sources(ag, ag.sources_bar_admission, ag.sources_general),
        additional_instruction=(
            "Verify the bar admission year as 2006 and that the admission pertains specifically to the AG's current state bar."
        )
    )

    # 2.g) Undergrad_From_In_State_University
    undergrad_leaf = evaluator.add_leaf(
        id="Undergrad_From_In_State_University",
        desc="Earned their undergraduate degree from a university located in the same state where they currently serve as Attorney General.",
        parent=criteria_node,
        critical=True
    )
    undergrad_claim = (
        f"{_nz(ag.name)} earned their undergraduate degree from '{_nz(ag.undergrad_university_name)}', "
        f"located in {_nz(ag.undergrad_university_state)}, the same state as {_nz(ag.state)}."
    )
    await evaluator.verify(
        claim=undergrad_claim,
        node=undergrad_leaf,
        sources=_pick_sources(ag, ag.sources_undergrad, ag.sources_general),
        additional_instruction="Confirm both the undergraduate institution and that it is located in the same state as the AG's current state."
    )

    # 2.h) At_Least_15_Years_Private_Practice
    practice_leaf = evaluator.add_leaf(
        id="At_Least_15_Years_Private_Practice",
        desc="Spent at least 15 years in private legal practice before becoming Attorney General.",
        parent=criteria_node,
        critical=True
    )
    practice_claim = (
        f"Before becoming Attorney General, {_nz(ag.name)} spent at least 15 years in private legal practice."
    )
    await evaluator.verify(
        claim=practice_claim,
        node=practice_leaf,
        sources=_pick_sources(ag, ag.sources_private_practice, ag.sources_general),
        additional_instruction=(
            "Use the sources to assess private practice duration. If the total private practice time (from roles, dates, resumes) "
            "is 15 or more years prior to assuming office as AG, mark supported; otherwise not supported."
        )
    )

    # 3) Provide required educational information (critical, parallel)
    edu_node = evaluator.add_parallel(
        id="Provide_Required_Educational_Information",
        desc="Provide the required educational institution names for the identified Attorney General.",
        parent=main_task,
        critical=True
    )

    # 3.a) Provide_Law_School_Name
    law_school_leaf = evaluator.add_leaf(
        id="Provide_Law_School_Name",
        desc="Provide the name of the law school where they earned their Juris Doctor.",
        parent=edu_node,
        critical=True
    )
    law_school_claim = f"The name of the law school (JD) is '{_nz(ag.law_school_name)}'."
    await evaluator.verify(
        claim=law_school_claim,
        node=law_school_leaf,
        sources=_pick_sources(ag, ag.sources_jd, ag.sources_general),
        additional_instruction="Confirm the JD-granting law school's name using the provided sources."
    )

    # 3.b) Provide_Undergraduate_University_Name
    undergrad_name_leaf = evaluator.add_leaf(
        id="Provide_Undergraduate_University_Name",
        desc="Provide the name of the university where they earned their undergraduate degree.",
        parent=edu_node,
        critical=True
    )
    undergrad_name_claim = f"The name of the undergraduate university is '{_nz(ag.undergrad_university_name)}'."
    await evaluator.verify(
        claim=undergrad_name_claim,
        node=undergrad_name_leaf,
        sources=_pick_sources(ag, ag.sources_undergrad, ag.sources_general),
        additional_instruction="Confirm the undergraduate university's name using the provided sources."
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
    Evaluate an agent's answer for the US state Attorney General criteria task.
    Returns a structured summary with verification tree and final score.
    """
    # Initialize evaluator (root is non-critical; we add the critical main task node under it)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # The overall task is sequential per rubric
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

    # Extract structured info from the answer
    ag_info: AGInfo = await evaluator.extract(
        prompt=prompt_extract_ag_info(),
        template_class=AGInfo,
        extraction_name="ag_info_extraction",
    )

    # Add helpful custom info (date range context) for transparency
    evaluator.add_custom_info(
        info={"date_range_start": DATE_RANGE_START, "date_range_end": DATE_RANGE_END},
        info_type="context",
        info_name="office_date_range_context"
    )

    # Build tree and run verifications
    await build_verification_tree_and_run(evaluator, ag_info)

    # Return evaluation summary
    return evaluator.get_summary()