import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "coach_career_investigation"
TASK_DESCRIPTION = (
    "I'm researching the career trajectory of college basketball coaches who have successfully progressed through multiple levels of competition. "
    "I need to identify a specific coach who meets all of the following criteria: "
    "(1) Currently serves as the head coach of a NCAA Division I men's basketball program, "
    "(2) Was hired to their current position in 2023, "
    "(3) At their immediately previous head coaching position, won the Atlantic 10 Conference Coach of the Year award in 2019, "
    "(4) In that same 2019 season at that position, their team also won the Atlantic 10 regular season championship, "
    "(5) Before their most recent head coaching position, served as a head coach at Rice University, "
    "(6) During their own playing career, competed in basketball at a NCAA Division III institution, "
    "(7) While playing, their Division III team won the NCAA Division III National Championship, "
    "(8) Earned an undergraduate degree in history, "
    "(9) Earned a master's degree from a university where they later returned to work as a coach. "
    "Please identify this coach and provide the following information: the coach's full name, their current institution and position, "
    "the institution where they won the 2019 Atlantic 10 Coach of the Year award, the Division III institution where they played and won a national championship, "
    "and the institution where they earned their master's degree and also later coached. "
    "For each piece of information, include reference URLs from reliable sources that verify these facts."
)

# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class CoachExtraction(BaseModel):
    # Identity
    identified_coach_name: Optional[str] = None
    coach_identity_urls: List[str] = Field(default_factory=list)
    multiple_final_candidates: Optional[bool] = None  # True if the answer suggests more than one final candidate

    # Current position (D1 men's head coach)
    current_institution: Optional[str] = None
    current_position: Optional[str] = None
    current_urls: List[str] = Field(default_factory=list)

    # Hired to current position in 2023
    hired_year: Optional[str] = None  # Keep as string to be lenient (e.g., "2023", "March 2023")
    hired_urls: List[str] = Field(default_factory=list)

    # Immediately previous head-coaching institution
    previous_hc_institution: Optional[str] = None
    previous_hc_urls: List[str] = Field(default_factory=list)

    # A-10 Coach of the Year 2019 at previous institution
    a10_coy_2019_institution: Optional[str] = None
    a10_coy_2019_urls: List[str] = Field(default_factory=list)

    # A-10 regular season championship 2019 (at same previous position)
    a10_regular_season_2019_institution: Optional[str] = None
    a10_regular_season_2019_urls: List[str] = Field(default_factory=list)

    # Head coach at Rice University prior to the most recent/immediately previous position
    rice_hc_before_previous: Optional[bool] = None
    rice_hc_urls: List[str] = Field(default_factory=list)

    # Division III playing career and DIII national title
    d3_playing_institution: Optional[str] = None
    d3_championship_year: Optional[str] = None  # optional
    d3_urls: List[str] = Field(default_factory=list)

    # Undergraduate degree in history
    undergrad_field: Optional[str] = None
    undergrad_institution: Optional[str] = None
    undergrad_urls: List[str] = Field(default_factory=list)

    # Master's degree and later coached at the same institution
    masters_institution: Optional[str] = None
    masters_field: Optional[str] = None
    masters_urls: List[str] = Field(default_factory=list)
    masters_later_coached_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_profile() -> str:
    return """
    Extract the single coach explicitly identified in the answer as the final answer who satisfies all listed constraints.
    You must only extract information that is explicitly present in the answer text and the URLs explicitly cited by the answer.

    Provide the following fields (use null for any missing field; for all *_urls fields, return an array of the explicit URLs cited in the answer for that fact; do not invent any URL):

    1) identified_coach_name: The full name of the identified coach (the one asserted to satisfy all criteria).
    2) coach_identity_urls: URL(s) cited to support the identity of this coach.

    3) current_institution: The current institution (university) where the coach is the head coach of men's basketball.
    4) current_position: The current position title (e.g., "Head Coach, Men's Basketball").
    5) current_urls: URL(s) cited that support the current role and institution (ideally official site, major news, or Wikipedia).

    6) hired_year: The year the coach was hired to the current position (must be explicitly given; use a 4-digit string like "2023" if possible).
    7) hired_urls: URL(s) cited that support the hire date/year.

    8) previous_hc_institution: The immediately previous head coaching institution (the head coach job held right before the current job).
    9) previous_hc_urls: URL(s) cited for the immediately previous head-coaching position.

    10) a10_coy_2019_institution: The institution where the coach won the Atlantic 10 Conference Coach of the Year award in 2019.
    11) a10_coy_2019_urls: URL(s) cited to support that the coach won the A-10 Coach of the Year in 2019 at that institution.

    12) a10_regular_season_2019_institution: The institution whose team won the Atlantic 10 regular season championship in the 2019 season under this coach.
    13) a10_regular_season_2019_urls: URL(s) cited to support that 2019 A-10 regular season championship.

    14) rice_hc_before_previous: true/false indicating whether the coach also served as head coach at Rice University prior to the immediately previous head coaching position.
    15) rice_hc_urls: URL(s) cited to confirm head-coaching tenure at Rice University prior to the immediately previous job.

    16) d3_playing_institution: The NCAA Division III institution where the coach played basketball.
    17) d3_championship_year: The year the coach's team won the NCAA Division III National Championship (if mentioned).
    18) d3_urls: URL(s) cited to support their Division III playing career and the DIII national title as a player.

    19) undergrad_field: The field of the undergraduate degree (e.g., "history").
    20) undergrad_institution: The institution that granted the undergraduate degree.
    21) undergrad_urls: URL(s) cited to support the undergraduate degree in history.

    22) masters_institution: The institution that granted the coach's master's degree.
    23) masters_field: The master's degree field/discipline (if mentioned).
    24) masters_urls: URL(s) cited to support the master's degree.
    25) masters_later_coached_urls: URL(s) cited to support that the coach later returned to the same master's-degree institution to work as a coach.

    26) multiple_final_candidates: true/false — Is the answer presenting more than one candidate coach as the final answer? 
        Set this to true ONLY if the answer explicitly proposes multiple different people as the final identified coach.

    SPECIAL RULES:
    - Only include URLs that are explicitly present in the answer text. Extract full URLs (markdown links are okay; extract the URL portion).
    - Do not infer or add any URLs not in the answer.
    - If a field is not explicitly stated, set it to null. If URLs are not provided for a field, return an empty array for that *_urls field.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls or []:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _first_non_empty(*url_lists: List[str]) -> List[str]:
    for lst in url_lists:
        if lst and len([u for u in lst if isinstance(u, str) and u.strip()]) > 0:
            return _dedup_urls(lst)
    return []


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_coach_profile(evaluator: Evaluator, parent_node, data: CoachExtraction) -> None:
    """
    Build verification leaves according to the rubric and run claim checks.
    All children under this node are critical.
    """
    # Create the main critical parallel node as described by the rubric
    main = evaluator.add_parallel(
        id="Coach_Career_Investigation",
        desc="Identify a single coach meeting all constraints and provide all requested fields, each supported by reliable reference URL(s).",
        parent=parent_node,
        critical=True,
    )

    # 1) Single_Coach_Identified (custom boolean check)
    single_ok = bool(data.identified_coach_name and str(data.identified_coach_name).strip()) and (data.multiple_final_candidates is False)
    evaluator.add_custom_node(
        result=single_ok,
        id="Single_Coach_Identified",
        desc="Response identifies exactly one coach as the answer (not multiple candidates).",
        parent=main,
        critical=True
    )

    # Helper variables
    name = (data.identified_coach_name or "").strip()
    current_inst = (data.current_institution or "").strip()
    prev_inst = (data.previous_hc_institution or "").strip()
    a10_coy_inst = (data.a10_coy_2019_institution or "").strip()
    a10_reg_inst = (data.a10_regular_season_2019_institution or "").strip()
    d3_inst = (data.d3_playing_institution or "").strip()
    masters_inst = (data.masters_institution or "").strip()

    # 2) Coach_Full_Name_With_Citation
    node_fullname = evaluator.add_leaf(
        id="Coach_Full_Name_With_Citation",
        desc="Response provides the coach's full name and includes a reliable reference URL supporting the identity.",
        parent=main,
        critical=True
    )
    identity_sources = _first_non_empty(data.coach_identity_urls, data.current_urls, data.previous_hc_urls)
    await evaluator.verify(
        claim=f"The identified coach is '{name}', as supported by the provided source page(s).",
        node=node_fullname,
        sources=identity_sources,
        additional_instruction="Verify that the source(s) clearly identify the person by the stated full name as a men's college basketball coach or a relevant biography page. Minor variations (middle initial, accents) are acceptable."
    )

    # 3) Current_Position_D1_Mens_Head_Coach_With_Citation
    node_current = evaluator.add_leaf(
        id="Current_Position_D1_Mens_Head_Coach_With_Citation",
        desc="Response states the coach's current institution and that they are the head coach of an NCAA Division I men's basketball program, with reliable reference URL(s).",
        parent=main,
        critical=True
    )
    curr_sources = _first_non_empty(data.current_urls, data.hired_urls)
    await evaluator.verify(
        claim=f"{name} is the head coach of the men's basketball team at {current_inst}, which competes in NCAA Division I.",
        node=node_current,
        sources=curr_sources,
        additional_instruction="Confirm both: (1) the person is the men's basketball head coach at the stated institution, and (2) that program is in NCAA Division I."
    )

    # 4) Hired_In_2023_With_Citation
    node_hired = evaluator.add_leaf(
        id="Hired_In_2023_With_Citation",
        desc="Response states the coach was hired to the current position in 2023, with reliable reference URL(s).",
        parent=main,
        critical=True
    )
    hire_sources = _first_non_empty(data.hired_urls, data.current_urls)
    await evaluator.verify(
        claim=f"{name} was hired in 2023 for the head coach position at {current_inst}.",
        node=node_hired,
        sources=hire_sources,
        additional_instruction="Check for explicit mention of the year 2023 in the hiring announcement or biography for the current head-coach position."
    )

    # 5) Immediately_Previous_HC_Institution_With_Citation
    node_prev = evaluator.add_leaf(
        id="Immediately_Previous_HC_Institution_With_Citation",
        desc="Response states the institution of the coach's immediately previous head coaching position (the position held directly before the current job), with reliable reference URL(s).",
        parent=main,
        critical=True
    )
    prev_sources = _first_non_empty(data.previous_hc_urls, data.a10_coy_2019_urls)
    await evaluator.verify(
        claim=f"Before taking the current job at {current_inst}, {name}'s immediately previous head-coaching position was at {prev_inst}.",
        node=node_prev,
        sources=prev_sources,
        additional_instruction="Verify that this institution was indeed the immediately previous HEAD coaching job (not an assistant role) directly before the current role."
    )

    # 6) Won_A10_Coach_of_Year_2019_Previous_Position_With_Citation
    node_a10_coy = evaluator.add_leaf(
        id="Won_A10_Coach_of_Year_2019_Previous_Position_With_Citation",
        desc="Response states that at the immediately previous head coaching position the coach won the Atlantic 10 Conference Coach of the Year award in 2019, with reliable reference URL(s).",
        parent=main,
        critical=True
    )
    a10_coy_sources = _first_non_empty(data.a10_coy_2019_urls, data.previous_hc_urls)
    await evaluator.verify(
        claim=f"In 2019, while head coach at {a10_coy_inst}, {name} won the Atlantic 10 Conference Coach of the Year award.",
        node=node_a10_coy,
        sources=a10_coy_sources,
        additional_instruction="The source(s) must explicitly indicate that the person won the A-10 Coach of the Year in 2019 and that it corresponds to the immediately previous head-coaching position."
    )

    # 7) Team_Won_A10_Regular_Season_Championship_2019_With_Citation
    node_a10_reg = evaluator.add_leaf(
        id="Team_Won_A10_Regular_Season_Championship_2019_With_Citation",
        desc="Response states that in the 2019 season at that same (immediately previous) head coaching position, the coach's team won the Atlantic 10 regular season championship, with reliable reference URL(s).",
        parent=main,
        critical=True
    )
    a10_reg_sources = _first_non_empty(data.a10_regular_season_2019_urls, data.a10_coy_2019_urls)
    await evaluator.verify(
        claim=f"In the 2019 season at {a10_reg_inst}, the team won the Atlantic 10 regular season championship under {name}.",
        node=node_a10_reg,
        sources=a10_reg_sources,
        additional_instruction="The source(s) should clearly affirm the A-10 regular-season title for the 2019 season for that institution."
    )

    # 8) Rice_Head_Coach_Before_Most_Recent_With_Citation
    node_rice = evaluator.add_leaf(
        id="Rice_Head_Coach_Before_Most_Recent_With_Citation",
        desc="Response states the coach served as head coach at Rice University before the most recent/immediately previous head coaching position, with reliable reference URL(s).",
        parent=main,
        critical=True
    )
    rice_sources = _first_non_empty(data.rice_hc_urls, data.previous_hc_urls)
    await evaluator.verify(
        claim=f"Before the immediately previous head-coaching position at {prev_inst}, {name} served as head coach at Rice University.",
        node=node_rice,
        sources=rice_sources,
        additional_instruction="Confirm that Rice University head-coaching tenure preceded the immediately previous head-coaching job."
    )

    # 9) D3_Playing_Institution_With_Citation
    node_d3_inst = evaluator.add_leaf(
        id="D3_Playing_Institution_With_Citation",
        desc="Response states the NCAA Division III institution where the coach played basketball, with reliable reference URL(s).",
        parent=main,
        critical=True
    )
    d3_sources = _first_non_empty(data.d3_urls)
    await evaluator.verify(
        claim=f"During his playing career, {name} competed in basketball at the NCAA Division III institution {d3_inst}.",
        node=node_d3_inst,
        sources=d3_sources,
        additional_instruction="Verify the Division III status and that the individual actually played for that school's men's basketball team."
    )

    # 10) D3_National_Championship_As_Player_With_Citation
    node_d3_title = evaluator.add_leaf(
        id="D3_National_Championship_As_Player_With_Citation",
        desc="Response states the coach's Division III team won the NCAA Division III National Championship during their playing career, with reliable reference URL(s).",
        parent=main,
        critical=True
    )
    if data.d3_championship_year and str(data.d3_championship_year).strip():
        d3_claim = f"While playing at {d3_inst}, {name}'s team won the NCAA Division III National Championship in {data.d3_championship_year}."
    else:
        d3_claim = f"While playing at {d3_inst}, {name}'s team won the NCAA Division III National Championship."
    await evaluator.verify(
        claim=d3_claim,
        node=node_d3_title,
        sources=d3_sources,
        additional_instruction="The source(s) should clearly indicate that the team won the Division III national championship while the person was a player."
    )

    # 11) Undergraduate_Degree_History_With_Citation
    node_undergrad = evaluator.add_leaf(
        id="Undergraduate_Degree_History_With_Citation",
        desc="Response states the coach earned an undergraduate degree in history, with reliable reference URL(s).",
        parent=main,
        critical=True
    )
    undergrad_sources = _first_non_empty(data.undergrad_urls)
    await evaluator.verify(
        claim=f"{name} earned an undergraduate degree in history.",
        node=node_undergrad,
        sources=undergrad_sources,
        additional_instruction="Verify that the undergraduate field explicitly states 'history' or equivalent wording."
    )

    # 12) Masters_Degree_And_Later_Coached_Same_Institution_With_Citation
    node_masters = evaluator.add_leaf(
        id="Masters_Degree_And_Later_Coached_Same_Institution_With_Citation",
        desc="Response states the institution where the coach earned their master's degree and that they later returned to coach at that same institution, with reliable reference URL(s).",
        parent=main,
        critical=True
    )
    masters_sources_combined = _dedup_urls((data.masters_urls or []) + (data.masters_later_coached_urls or []))
    await evaluator.verify(
        claim=f"{name} earned a master's degree from {masters_inst} and later returned to that same institution to work as a coach.",
        node=node_masters,
        sources=masters_sources_combined,
        additional_instruction="The source(s) should support both parts: the master's degree from the named institution and that the person later coached there."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict[str, Any]:
    """
    Entry point to evaluate an answer for the coach career investigation task.
    """
    # Initialize evaluator with a PARALLEL root; use task description as root desc
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

    # Extraction
    extracted: CoachExtraction = await evaluator.extract(
        prompt=prompt_extract_coach_profile(),
        template_class=CoachExtraction,
        extraction_name="coach_profile_extraction"
    )

    # Build tree and run verifications
    await verify_coach_profile(evaluator, root, extracted)

    # Return standardized summary
    return evaluator.get_summary()