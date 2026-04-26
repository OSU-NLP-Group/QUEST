import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ninth_circuit_states_2026"
TASK_DESCRIPTION = (
    "Among the nine western states covered by the United States Court of Appeals for the Ninth Circuit, "
    "identify the four (4) states that had regular legislative sessions scheduled to begin in January 2026. "
    "For each of these four states, compile a comprehensive profile of their government transparency and procedural requirements, including: "
    "(1) The exact start and end dates of their 2026 regular legislative session; "
    "(2) The minimum advance notice period required by state law for open government meetings (specify the time period in hours or days); "
    "(3) The maximum response timeframe allowed under state law for responding to public records requests (specify in business days or calendar days); "
    "(4) The vote threshold required by state law or constitution to override a gubernatorial veto (express as a fraction or percentage). "
    "Additionally, for federal baseline comparison, provide the following federal government procedural requirements: "
    "(5) The minimum public comment period required under the Administrative Procedure Act (APA) for federal agency rulemaking; "
    "(6) The minimum advance notice period required by the Federal Advisory Committee Act (FACA) for advisory committee meetings published in the Federal Register; "
    "(7) The minimum number of days after Federal Register publication that federal rules must wait before taking effect under the APA; "
    "(8) The vote threshold required for the U.S. Congress to override a presidential veto. "
    "For all state and federal requirements, provide authoritative source URLs that verify each specific requirement."
)

NINTH_CIRCUIT_STATES = [
    "Alaska",
    "Arizona",
    "California",
    "Hawaii",
    "Idaho",
    "Montana",
    "Nevada",
    "Oregon",
    "Washington",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateProfile(BaseModel):
    state: Optional[str] = None

    session_start_date: Optional[str] = None
    session_end_date: Optional[str] = None
    session_sources: List[str] = Field(default_factory=list)

    open_meetings_notice_minimum: Optional[str] = None
    open_meetings_sources: List[str] = Field(default_factory=list)

    public_records_response_time_max: Optional[str] = None
    public_records_sources: List[str] = Field(default_factory=list)

    veto_override_threshold: Optional[str] = None
    veto_override_sources: List[str] = Field(default_factory=list)


class StatesAndProfilesExtraction(BaseModel):
    qualifying_states: List[str] = Field(default_factory=list)
    profiles: List[StateProfile] = Field(default_factory=list)


class FederalRequirementsExtraction(BaseModel):
    apa_comment_minimum: Optional[str] = None
    apa_comment_sources: List[str] = Field(default_factory=list)

    faca_notice_minimum: Optional[str] = None
    faca_notice_sources: List[str] = Field(default_factory=list)

    apa_effective_delay_minimum: Optional[str] = None
    apa_effective_delay_sources: List[str] = Field(default_factory=list)

    congress_veto_override_threshold: Optional[str] = None
    congress_veto_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_states_and_profiles() -> str:
    return (
        "Extract from the answer the four qualifying Ninth Circuit states that the answer claims have 2026 regular "
        "legislative sessions scheduled to begin in January 2026, and provide detailed profiles for each. "
        "Rules:\n"
        "- Ninth Circuit states are: Alaska, Arizona, California, Hawaii, Idaho, Montana, Nevada, Oregon, Washington.\n"
        "- qualifying_states: list exactly the states the answer claims meet the January 2026 start criterion; "
        "if more than four are listed, include only the first four mentioned; if fewer are listed, include what is present.\n"
        "- profiles: extract up to four profiles corresponding to these qualifying states. Each profile must include:\n"
        "  • state: full state name as labeled in the answer.\n"
        "  • session_start_date and session_end_date for the 2026 regular session.\n"
        "  • session_sources: authoritative URL(s) (e.g., official legislature site, statute, or government calendar) supporting the session dates.\n"
        "  • open_meetings_notice_minimum: the minimum advance public notice period for open meetings (hours/days).\n"
        "  • open_meetings_sources: authoritative URL(s) supporting the open meetings requirement.\n"
        "  • public_records_response_time_max: the maximum response timeframe for public records requests (business/calendar days).\n"
        "  • public_records_sources: authoritative URL(s) supporting the public records timeframe.\n"
        "  • veto_override_threshold: the vote threshold to override a gubernatorial veto (fraction/percentage).\n"
        "  • veto_override_sources: authoritative URL(s) supporting the veto override threshold.\n"
        "Source extraction rules:\n"
        "- Extract only URLs explicitly present in the answer. Do not invent any URLs.\n"
        "- Include full URLs, including protocol. If a URL lacks protocol, prepend http://.\n"
        "- If a required value is missing, set it to null. If sources are missing for an item, return an empty list.\n"
        "Return JSON with fields: qualifying_states (array of strings) and profiles (array of objects as specified)."
    )


def prompt_extract_federal_requirements() -> str:
    return (
        "Extract from the answer the federal baseline procedural requirements and their authoritative sources. "
        "For each item below, extract both the value and a list of authoritative source URL(s):\n"
        "1) apa_comment_minimum: The minimum public comment period under the Administrative Procedure Act (APA) for agency rulemaking.\n"
        "   • apa_comment_sources: URL(s) supporting the stated minimum.\n"
        "2) faca_notice_minimum: The minimum advance notice period under FACA for advisory committee meetings published in the Federal Register.\n"
        "   • faca_notice_sources: URL(s) supporting the stated minimum.\n"
        "3) apa_effective_delay_minimum: The minimum number of days after Federal Register publication before rules take effect under the APA.\n"
        "   • apa_effective_delay_sources: URL(s) supporting the stated minimum.\n"
        "4) congress_veto_override_threshold: The vote threshold for Congress to override a presidential veto.\n"
        "   • congress_veto_sources: URL(s) supporting the stated threshold.\n"
        "Source extraction rules:\n"
        "- Extract only URLs explicitly present in the answer. Do not invent any URLs.\n"
        "- Include full URLs, including protocol. If a URL lacks protocol, prepend http://.\n"
        "- If a required value is missing, set it to null. If sources are missing, return an empty list.\n"
        "Return a JSON object with these fields exactly."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _format_states_list(states: List[str]) -> str:
    return ", ".join([s for s in states if s])


def _is_nonempty_string(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _has_sources(urls: List[str]) -> bool:
    return bool(urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_identify_qualifying_states(
    evaluator: Evaluator,
    parent_node,
    extracted: StatesAndProfilesExtraction,
) -> None:
    """
    Add a leaf node to verify that the answer listed exactly four distinct qualifying Ninth Circuit states
    and claimed they have regular sessions beginning in January 2026.
    This is a logical consistency check based on the answer text (simple verification).
    """
    leaf = evaluator.add_leaf(
        id="Identify_Qualifying_States",
        desc=(
            "Lists exactly four distinct states that are among the Ninth Circuit states and that have a 2026 regular legislative "
            "session scheduled to begin in January 2026."
        ),
        parent=parent_node,
        critical=True,
    )

    stated_states = extracted.qualifying_states[:4]
    claim = (
        "The answer identifies exactly four distinct qualifying states among the Ninth Circuit states, and each of these "
        "has a 2026 regular legislative session scheduled to begin in January 2026. "
        f"The four states listed are: {_format_states_list(stated_states)}. "
        f"The Ninth Circuit states are: {_format_states_list(NINTH_CIRCUIT_STATES)}. "
        "Check that the count is exactly four, the states are distinct, and all belong to the Ninth Circuit set."
    )

    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction=(
            "Use the task description to validate the Ninth Circuit roster. This check is performed against the answer's own content. "
            "Confirm there are exactly four distinct states, all within the Ninth Circuit. "
            "Do not rely on external sources for this specific logical count/membership check; subsequent nodes will verify dates via sources."
        ),
    )


async def verify_state_profile(
    evaluator: Evaluator,
    parent_node,
    profile: StateProfile,
    idx: int,
    qualifying_states_set: set,
) -> None:
    """
    Build and verify the four core attributes (with sources) for a given state profile.
    Also include a preliminary label presence/consistency check as a critical custom node.
    """
    # State profile node (must be critical because parent is critical per rubric)
    profile_node = evaluator.add_parallel(
        id=f"State_Profile_{idx + 1}",
        desc="Profile for one of the identified qualifying states.",
        parent=parent_node,
        critical=True,
    )

    # 0) Label presence and consistency with identified states
    label_ok = _is_nonempty_string(profile.state) and (profile.state in NINTH_CIRCUIT_STATES) and (
        profile.state in qualifying_states_set if qualifying_states_set else True
    )
    evaluator.add_custom_node(
        result=label_ok,
        id=f"State_Profile_{idx + 1}_Label_Valid",
        desc="Profile state label is present, is a Ninth Circuit state, and corresponds to one of the identified qualifying states.",
        parent=profile_node,
        critical=True,
    )

    # 1) Session dates with sources — existence prerequisite
    session_sources_present = _has_sources(profile.session_sources) and _is_nonempty_string(profile.session_start_date) and _is_nonempty_string(profile.session_end_date)
    session_sources_node = evaluator.add_custom_node(
        result=session_sources_present,
        id=f"Session_Dates_Sources_Present_{idx + 1}",
        desc="Session dates value(s) present and authoritative source URL(s) provided.",
        parent=profile_node,
        critical=True,
    )

    session_leaf = evaluator.add_leaf(
        id=f"Session_Dates_With_Source_{idx + 1}",
        desc="Gives the exact start date and end date of the state's 2026 regular legislative session, with authoritative source URL(s).",
        parent=profile_node,
        critical=True,
    )

    session_claim = (
        f"For {profile.state}, the 2026 regular legislative session runs from {profile.session_start_date} to {profile.session_end_date}."
    )
    await evaluator.verify(
        claim=session_claim,
        node=session_leaf,
        sources=profile.session_sources,
        additional_instruction=(
            "Verify the exact 2026 regular session start and end dates using authoritative calendar or legislative sources. "
            "If multiple phases or sessions exist, focus on the 2026 regular session. Accept common date formatting variants."
        ),
        extra_prerequisites=[session_sources_node],
    )

    # 2) Open meetings notice minimum — existence prerequisite
    om_sources_present = _has_sources(profile.open_meetings_sources) and _is_nonempty_string(profile.open_meetings_notice_minimum)
    om_sources_node = evaluator.add_custom_node(
        result=om_sources_present,
        id=f"Open_Meetings_Sources_Present_{idx + 1}",
        desc="Open meetings notice minimum value present and authoritative source URL(s) provided.",
        parent=profile_node,
        critical=True,
    )

    om_leaf = evaluator.add_leaf(
        id=f"Open_Meetings_Notice_With_Source_{idx + 1}",
        desc="Gives the minimum advance public notice period required by state law for open government meetings (hours/days), with authoritative source URL(s).",
        parent=profile_node,
        critical=True,
    )

    om_claim = (
        f"In {profile.state}, the minimum advance public notice period required for open government meetings is {profile.open_meetings_notice_minimum}."
    )
    await evaluator.verify(
        claim=om_claim,
        node=om_leaf,
        sources=profile.open_meetings_sources,
        additional_instruction=(
            "Confirm the minimum notice period from the state's open meetings/sunshine law or equivalent authoritative source. "
            "Value may be in hours or days; verify the minimum requirement."
        ),
        extra_prerequisites=[om_sources_node],
    )

    # 3) Public records response timeframe — existence prerequisite
    pr_sources_present = _has_sources(profile.public_records_sources) and _is_nonempty_string(profile.public_records_response_time_max)
    pr_sources_node = evaluator.add_custom_node(
        result=pr_sources_present,
        id=f"Public_Records_Sources_Present_{idx + 1}",
        desc="Public records response timeframe value present and authoritative source URL(s) provided.",
        parent=profile_node,
        critical=True,
    )

    pr_leaf = evaluator.add_leaf(
        id=f"Public_Records_Response_With_Source_{idx + 1}",
        desc="Gives the maximum response timeframe allowed under state law for public records requests (business/calendar days), with authoritative source URL(s).",
        parent=profile_node,
        critical=True,
    )

    pr_claim = (
        f"In {profile.state}, the maximum response timeframe under state law for public records requests is {profile.public_records_response_time_max}."
    )
    await evaluator.verify(
        claim=pr_claim,
        node=pr_leaf,
        sources=profile.public_records_sources,
        additional_instruction=(
            "Verify the stated maximum response time from authoritative sources (statute, administrative rule, official guidance). "
            "Clarify whether days are business or calendar; accept reasonable phrasing variations."
        ),
        extra_prerequisites=[pr_sources_node],
    )

    # 4) Veto override threshold — existence prerequisite
    vo_sources_present = _has_sources(profile.veto_override_sources) and _is_nonempty_string(profile.veto_override_threshold)
    vo_sources_node = evaluator.add_custom_node(
        result=vo_sources_present,
        id=f"Veto_Override_Sources_Present_{idx + 1}",
        desc="Veto override threshold value present and authoritative source URL(s) provided.",
        parent=profile_node,
        critical=True,
    )

    vo_leaf = evaluator.add_leaf(
        id=f"Veto_Override_With_Source_{idx + 1}",
        desc="Gives the vote threshold required to override a gubernatorial veto (fraction/percentage), with authoritative source URL(s).",
        parent=profile_node,
        critical=True,
    )

    vo_claim = (
        f"In {profile.state}, overriding a gubernatorial veto requires {profile.veto_override_threshold}."
    )
    await evaluator.verify(
        claim=vo_claim,
        node=vo_leaf,
        sources=profile.veto_override_sources,
        additional_instruction=(
            "Confirm the override threshold from the state constitution or authoritative statute. "
            "Threshold may be expressed as two-thirds, three-fifths, or a percentage; allow equivalent descriptions."
        ),
        extra_prerequisites=[vo_sources_node],
    )


async def verify_federal_requirements(
    evaluator: Evaluator,
    parent_node,
    fed: FederalRequirementsExtraction,
) -> None:
    """
    Build the four federal baseline requirement checks with source verification.
    Include source existence prerequisites for each.
    """
    # APA comment period
    apa_comment_exist = _has_sources(fed.apa_comment_sources) and _is_nonempty_string(fed.apa_comment_minimum)
    apa_comment_exist_node = evaluator.add_custom_node(
        result=apa_comment_exist,
        id="APA_Comment_Period_Sources_Present",
        desc="APA comment period value present and authoritative source URL(s) provided.",
        parent=parent_node,
        critical=True,
    )

    apa_comment_leaf = evaluator.add_leaf(
        id="APA_Comment_Period_With_Source",
        desc="States the APA minimum public comment period for rulemaking, with authoritative source URL(s).",
        parent=parent_node,
        critical=True,
    )

    apa_comment_claim = (
        f"The minimum public comment period required under the APA for federal agency rulemaking is {fed.apa_comment_minimum}."
    )
    await evaluator.verify(
        claim=apa_comment_claim,
        node=apa_comment_leaf,
        sources=fed.apa_comment_sources,
        additional_instruction=(
            "Verify the stated minimum or baseline from authoritative sources (statute/regulation/official guidance). "
            "If typical practice differs, focus on the minimum requirement and note exceptions only if the source explicitly indicates them."
        ),
        extra_prerequisites=[apa_comment_exist_node],
    )

    # FACA notice period
    faca_notice_exist = _has_sources(fed.faca_notice_sources) and _is_nonempty_string(fed.faca_notice_minimum)
    faca_notice_exist_node = evaluator.add_custom_node(
        result=faca_notice_exist,
        id="FACA_Notice_Period_Sources_Present",
        desc="FACA notice period value present and authoritative source URL(s) provided.",
        parent=parent_node,
        critical=True,
    )

    faca_notice_leaf = evaluator.add_leaf(
        id="FACA_Notice_Period_With_Source",
        desc="States the FACA minimum advance notice period for advisory committee meetings in the Federal Register, with authoritative source URL(s).",
        parent=parent_node,
        critical=True,
    )

    faca_notice_claim = (
        f"The minimum advance notice period required under FACA for advisory committee meetings in the Federal Register is {fed.faca_notice_minimum}."
    )
    await evaluator.verify(
        claim=faca_notice_claim,
        node=faca_notice_leaf,
        sources=fed.faca_notice_sources,
        additional_instruction=(
            "Verify the notice period from authoritative sources (statute/regulation/official Federal Register guidance). "
            "Confirm that the stated minimum is consistent with governing requirements."
        ),
        extra_prerequisites=[faca_notice_exist_node],
    )

    # APA effective date delay
    apa_delay_exist = _has_sources(fed.apa_effective_delay_sources) and _is_nonempty_string(fed.apa_effective_delay_minimum)
    apa_delay_exist_node = evaluator.add_custom_node(
        result=apa_delay_exist,
        id="APA_Effective_Delay_Sources_Present",
        desc="APA effective date delay value present and authoritative source URL(s) provided.",
        parent=parent_node,
        critical=True,
    )

    apa_delay_leaf = evaluator.add_leaf(
        id="APA_Effective_Date_Delay_With_Source",
        desc="States the minimum delay after Federal Register publication before federal rules can take effect under the APA, with authoritative source URL(s).",
        parent=parent_node,
        critical=True,
    )

    apa_delay_claim = (
        f"The minimum number of days after Federal Register publication before federal rules can take effect under the APA is {fed.apa_effective_delay_minimum}."
    )
    await evaluator.verify(
        claim=apa_delay_claim,
        node=apa_delay_leaf,
        sources=fed.apa_effective_delay_sources,
        additional_instruction=(
            "Verify the effective date delay from authoritative sources (statute/regulation/official guidance). "
            "If exceptions exist, the source should indicate them; focus on the minimum delay requirement."
        ),
        extra_prerequisites=[apa_delay_exist_node],
    )

    # Congress veto override threshold
    cv_exist = _has_sources(fed.congress_veto_sources) and _is_nonempty_string(fed.congress_veto_override_threshold)
    cv_exist_node = evaluator.add_custom_node(
        result=cv_exist,
        id="Congress_Veto_Override_Sources_Present",
        desc="Congress veto override threshold value present and authoritative source URL(s) provided.",
        parent=parent_node,
        critical=True,
    )

    cv_leaf = evaluator.add_leaf(
        id="Congress_Veto_Override_With_Source",
        desc="States the vote threshold for Congress to override a presidential veto, with authoritative source URL(s).",
        parent=parent_node,
        critical=True,
    )

    cv_claim = (
        f"The vote threshold required for the U.S. Congress to override a presidential veto is {fed.congress_veto_override_threshold}."
    )
    await evaluator.verify(
        claim=cv_claim,
        node=cv_leaf,
        sources=fed.congress_veto_sources,
        additional_instruction=(
            "Verify the override threshold from authoritative sources (U.S. Constitution or official government sources). "
            "Threshold is typically two-thirds of both House and Senate; accept equivalent wording."
        ),
        extra_prerequisites=[cv_exist_node],
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
    Evaluate the provided answer against the Ninth Circuit state and federal baseline requirements rubric.
    """
    # 1) Initialize evaluator and create a critical wrapper root (to enforce no partial credit across main sections)
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

    # Critical wrapper node to enforce overall critical status
    root_critical = evaluator.add_parallel(
        id="Root",
        desc="Evaluate whether the response satisfies the state requirements and the federal baseline requirements.",
        parent=root,
        critical=True,
    )

    # 2) Extract structured info from the answer
    states_and_profiles = await evaluator.extract(
        prompt=prompt_extract_states_and_profiles(),
        template_class=StatesAndProfilesExtraction,
        extraction_name="states_and_profiles",
    )

    federal_requirements = await evaluator.extract(
        prompt=prompt_extract_federal_requirements(),
        template_class=FederalRequirementsExtraction,
        extraction_name="federal_requirements",
    )

    # Record Ninth Circuit roster for transparency
    evaluator.add_custom_info(
        info={"ninth_circuit_states": NINTH_CIRCUIT_STATES},
        info_type="context",
        info_name="Ninth Circuit States Roster",
    )

    # 3) Build State Requirements sub-tree (sequential, critical)
    state_requirements_node = evaluator.add_sequential(
        id="State_Requirements",
        desc="State portion: identify the four qualifying Ninth Circuit states and provide required attributes for each.",
        parent=root_critical,
        critical=True,
    )

    # 3.1) Identify qualifying states (leaf, critical)
    await verify_identify_qualifying_states(evaluator, state_requirements_node, states_and_profiles)

    # 3.2) Provide four state profiles (parallel, critical)
    provide_profiles_node = evaluator.add_parallel(
        id="Provide_Four_State_Profiles",
        desc="Provides four separate state profiles, each clearly labeled with its state, corresponding one-to-one to the four identified qualifying states (no duplicates, no omissions).",
        parent=state_requirements_node,
        critical=True,
    )

    # Prepare profiles: use first four profiles; pad with empties if fewer
    profiles: List[StateProfile] = states_and_profiles.profiles[:4]
    while len(profiles) < 4:
        profiles.append(StateProfile())

    qualifying_set = set(states_and_profiles.qualifying_states[:4]) if states_and_profiles.qualifying_states else set()

    # Verify each state profile
    for i in range(4):
        await verify_state_profile(
            evaluator=evaluator,
            parent_node=provide_profiles_node,
            profile=profiles[i],
            idx=i,
            qualifying_states_set=qualifying_set,
        )

    # 4) Build Federal Baseline Requirements sub-tree (parallel, critical)
    federal_node = evaluator.add_parallel(
        id="Federal_Baseline_Requirements",
        desc="Federal portion: provide the requested federal procedural requirements, each with authoritative source URL(s).",
        parent=root_critical,
        critical=True,
    )

    await verify_federal_requirements(evaluator, federal_node, federal_requirements)

    # 5) Return structured summary
    return evaluator.get_summary()