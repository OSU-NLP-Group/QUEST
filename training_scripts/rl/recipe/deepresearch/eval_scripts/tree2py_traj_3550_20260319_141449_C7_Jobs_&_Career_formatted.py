import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "mw_ad_selection_2026"
TASK_DESCRIPTION = """
Identify the current athletic director position at a Mountain West Conference institution that meets ALL of the following criteria as of March 2026:

1. The institution competes at the NCAA Division I Football Bowl Subdivision (FBS) level
2. The institution is a current member of the Mountain West Conference
3. The institution is located in a state in the western United States
4. The institution is a public university
5. The institution sponsors an FBS football program
6. The institution hired a new head football coach between January 2024 and March 2026
7. The athletic department has allocated approximately $20 million or more annually for direct student-athlete compensation
8. The athletic department sponsors at least 14 NCAA varsity sports
9. The athletic department is currently planning or undertaking significant stadium renovation or capital improvement projects
10. The institution has implemented or approved student fee increases to support athletics within the past two years (2024-2026)
11. The institution has sought or received state legislative funding for athletic facilities or stadium improvements in the current (2026) legislative session

Provide the name of the institution and the name of the current athletic director holding this position.
"""


# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class MWADExtraction(BaseModel):
    # Identity
    institution_name: Optional[str] = None
    institution_state: Optional[str] = None  # if available in the answer
    athletic_director_name: Optional[str] = None
    ad_profile_urls: List[str] = Field(default_factory=list)  # staff directory, bio page, news release confirming AD

    # MUST constraints evidence
    mw_membership_urls: List[str] = Field(default_factory=list)
    public_university_urls: List[str] = Field(default_factory=list)
    western_state_urls: List[str] = Field(default_factory=list)
    fbs_urls: List[str] = Field(default_factory=list)

    new_head_coach_urls: List[str] = Field(default_factory=list)

    comp_budget_amount: Optional[str] = None  # e.g., "about $22 million", ">= $20M"
    comp_budget_urls: List[str] = Field(default_factory=list)

    varsity_sports_count: Optional[str] = None  # e.g., "16" or "at least 14"
    varsity_sports_urls: List[str] = Field(default_factory=list)

    stadium_projects_desc: Optional[str] = None
    stadium_projects_urls: List[str] = Field(default_factory=list)

    student_fee_increase_desc: Optional[str] = None
    student_fee_increase_urls: List[str] = Field(default_factory=list)

    state_legislative_funding_desc: Optional[str] = None
    state_legislative_funding_urls: List[str] = Field(default_factory=list)

    ad_compliance_oversight_urls: List[str] = Field(default_factory=list)

    # SHOULD (non-critical) constraints evidence
    nil_program_desc: Optional[str] = None
    nil_program_urls: List[str] = Field(default_factory=list)

    nacda_membership_requirement_urls: List[str] = Field(default_factory=list)
    advanced_degree_requirement_urls: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_mw_ad_selection() -> str:
    return """
Extract, from the provided answer, ALL structured information about the selected Mountain West Conference (MWC) institution and its current athletic director (as of March 2026), along with URLs that the answer cites as evidence for each criterion. Follow these rules strictly:
- Extract only what is explicitly present in the answer text.
- For each 'urls' field, return only valid, complete URLs that appear in the answer (including within markdown links). Do not fabricate or infer URLs.
- If a field is missing, set it to null (for strings) or an empty list (for urls).

Required fields to extract (use these exact JSON keys):
1) identity
- institution_name: string or null
- institution_state: string or null (two-letter or full state name if available)
- athletic_director_name: string or null
- ad_profile_urls: list of URLs that directly confirm the current AD at the institution

2) MUST constraints (each with its own URL list):
- mw_membership_urls: list of URLs showing the institution is a current MWC member as of March 2026
- public_university_urls: list of URLs confirming it is a public university
- western_state_urls: list of URLs confirming the institution's U.S. state (for checking it is in the western U.S.)
- fbs_urls: list of URLs confirming it competes in NCAA Division I FBS football

- new_head_coach_urls: list of URLs confirming a new head football coach was hired between Jan 2024 and Mar 2026

- comp_budget_amount: string or null describing the approximate annual direct athlete compensation allocation (e.g., "about $22 million", ">= $20M")
- comp_budget_urls: list of URLs supporting that amount (revenue sharing / direct athlete compensation)

- varsity_sports_count: string or null (e.g., "16", "at least 14")
- varsity_sports_urls: list of URLs supporting the varsity sports count

- stadium_projects_desc: string or null describing significant stadium renovation or capital projects as of Mar 2026
- stadium_projects_urls: list of URLs supporting those projects

- student_fee_increase_desc: string or null describing student fee increases (2024–2026) to support athletics
- student_fee_increase_urls: list of URLs supporting the student fee increase

- state_legislative_funding_desc: string or null describing funding sought/received in the 2026 state legislative session for athletic facilities/stadium
- state_legislative_funding_urls: list of URLs supporting the 2026 legislative funding activity

- ad_compliance_oversight_urls: list of URLs confirming the AD oversees NCAA rules compliance and athlete eligibility

3) SHOULD (non-critical) constraints (if present in the answer):
- nil_program_desc: string or null describing NIL program or collective
- nil_program_urls: list of URLs supporting NIL program or collective

- nacda_membership_requirement_urls: list of URLs supporting that the AD position requires NACDA membership
- advanced_degree_requirement_urls: list of URLs supporting that the AD position requires or prefers a Master's degree or higher
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    # deduplicate while preserving order
    seen = set()
    result = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


async def _add_identity_checks(
    evaluator: Evaluator,
    parent,
    data: MWADExtraction,
) -> None:
    """
    Sequential identity checks:
    1) Institution name provided (critical)
    2) AD name provided (critical)
    3) AD current as of March 2026 supported by URLs (critical)
    """
    inst = (data.institution_name or "").strip()
    ad_name = (data.athletic_director_name or "").strip()
    ad_urls = _non_empty_urls(data.ad_profile_urls)

    seq = evaluator.add_sequential(
        id="identity_checks",
        desc="Identity checks: institution name, AD name, and AD current as of March 2026 supported by sources",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(inst),
        id="Institution_Name_Provided",
        desc="The name of the institution is provided.",
        parent=seq,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(ad_name),
        id="Athletic_Director_Name_Provided",
        desc="The name of the current athletic director at the identified institution (as of March 2026) is provided.",
        parent=seq,
        critical=True,
    )

    # Sources provided for AD verification
    evaluator.add_custom_node(
        result=bool(ad_urls),
        id="AD_Sources_Provided",
        desc="Source URLs are provided to confirm the current athletic director as of March 2026.",
        parent=seq,
        critical=True,
    )

    # Verify AD current as of March 2026
    ad_verify_leaf = evaluator.add_leaf(
        id="AD_Current_AsOf_Mar2026_Supported",
        desc="As of March 2026, the named individual is the current athletic director of the institution (source-supported).",
        parent=seq,
        critical=True,
    )

    ad_claim = f"As of March 2026, {ad_name} is the current athletic director of {inst}."
    await evaluator.verify(
        claim=ad_claim,
        node=ad_verify_leaf,
        sources=ad_urls,
        additional_instruction=(
            "Verify that the provided page(s) explicitly identify this person as the current/active athletic director "
            f"at {inst}, as of March 2026 (staff directory, press release, bio page, or official announcement). "
            "Prefer official university/athletics sites or reputable news sources."
        ),
    )


async def _add_criterion_seq_with_sources(
    evaluator: Evaluator,
    parent,
    node_id: str,
    title: str,
    urls: List[str],
    claim: str,
    add_ins: str,
    critical: bool = True,
) -> None:
    """
    Generic pattern: create a sequential mini-chain with:
      1) sources provided (custom, gates verification)
      2) verify claim by URLs
    """
    seq = evaluator.add_sequential(
        id=node_id,
        desc=title,
        parent=parent,
        critical=critical,
    )

    evaluator.add_custom_node(
        result=bool(_non_empty_urls(urls)),
        id=f"{node_id}_sources_provided",
        desc=f"Sources provided for: {title}",
        parent=seq,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc=title,
        parent=seq,
        critical=True,
    )

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_non_empty_urls(urls),
        additional_instruction=add_ins,
    )


# -----------------------------------------------------------------------------
# Main evaluation entry
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the Mountain West AD selection task (as of March 2026).
    """
    # Initialize evaluator/root
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
        prompt=prompt_extract_mw_ad_selection(),
        template_class=MWADExtraction,
        extraction_name="mw_ad_selection_extraction",
    )

    # Convenience variables
    inst = (extraction.institution_name or "").strip()
    ad_name = (extraction.athletic_director_name or "").strip()
    state = (extraction.institution_state or "").strip()

    # Build top-level groups:
    # - MUST group (critical): all required constraints must pass
    # - SHOULD group (non-critical): optional constraints
    must_group = evaluator.add_parallel(
        id="MUST_Constraints",
        desc="All MUST constraints satisfied (as of March 2026).",
        parent=root,
        critical=True,
    )
    should_group = evaluator.add_parallel(
        id="SHOULD_Constraints",
        desc="Non-critical (SHOULD) constraints (partial credit if present).",
        parent=root,
        critical=False,
    )

    # Identity checks (critical)
    await _add_identity_checks(evaluator, must_group, extraction)

    # 1) Mountain West membership as of March 2026 (critical)
    await _add_criterion_seq_with_sources(
        evaluator,
        must_group,
        "MW_Conference_Membership_AsOf_Mar2026",
        "The institution is a current member of the Mountain West Conference as of March 2026.",
        extraction.mw_membership_urls,
        claim=f"As of March 2026, {inst} is a current member of the Mountain West Conference (MWC).",
        add_ins="Confirm current MWC membership status specifically for the March 2026 timeframe (e.g., conference website, official releases, or reputable news).",
        critical=True,
    )

    # 2) Public university (critical)
    await _add_criterion_seq_with_sources(
        evaluator,
        must_group,
        "Public_University",
        "The institution is a public university.",
        extraction.public_university_urls,
        claim=f"{inst} is a public university.",
        add_ins="Look for explicit statements of public status (e.g., 'public university', 'public institution', state system membership).",
        critical=True,
    )

    # 3) Western U.S. state (critical)
    western_list = "AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY"
    western_claim = (
        f"{inst} is located in {state}, which is in the western United States."
        if state else f"{inst} is located in a state in the western United States."
    )
    await _add_criterion_seq_with_sources(
        evaluator,
        must_group,
        "Western_US_State",
        "The institution is located in a state in the western United States.",
        extraction.western_state_urls,
        claim=western_claim,
        add_ins=(
            "Western U.S. states for this task include: "
            f"{western_list}. Verify the institution's state and confirm it is in this list."
        ),
        critical=True,
    )

    # 4) NCAA Division I FBS football (critical)
    await _add_criterion_seq_with_sources(
        evaluator,
        must_group,
        "NCAA_DivisionI_FBS_Football",
        "The institution competes at the NCAA Division I Football Bowl Subdivision (FBS) level (sponsors an FBS football program).",
        extraction.fbs_urls,
        claim=f"{inst} competes in NCAA Division I FBS (sponsors an FBS football program).",
        add_ins="Confirm FBS status explicitly; pages that list 'FBS' or membership in an FBS conference are acceptable.",
        critical=True,
    )

    # 5) New head football coach between Jan 2024 and Mar 2026 (critical)
    await _add_criterion_seq_with_sources(
        evaluator,
        must_group,
        "New_Head_Football_Coach_Jan2024_to_Mar2026",
        "The institution hired a new head football coach between January 2024 and March 2026.",
        extraction.new_head_coach_urls,
        claim=f"Between January 2024 and March 2026, {inst} hired a new head football coach.",
        add_ins="Verify the head coach hire date falls within Jan 1, 2024 through Mar 31, 2026 (inclusive).",
        critical=True,
    )

    # 6) Direct student-athlete compensation budget >= $20M (critical)
    budget_amt = (extraction.comp_budget_amount or "").strip()
    budget_title = "The athletic department has allocated ≈$20M or more annually for direct student-athlete compensation."
    budget_claim = (
        f"{inst}'s athletic department has allocated approximately $20 million or more annually for direct student-athlete compensation."
    )
    if budget_amt:
        budget_claim += f" The answer cites the amount as: {budget_amt}."

    await _add_criterion_seq_with_sources(
        evaluator,
        must_group,
        "Direct_Athlete_Compensation_Budget_Gte_20M",
        budget_title,
        extraction.comp_budget_urls,
        claim=budget_claim,
        add_ins=(
            "Accept phrasing like 'direct athlete compensation', 'revenue sharing', 'House settlement/student-athlete payments', "
            "'share-of-revenue' so long as the annual amount is approximately $20M or more. Reasonable approximations are acceptable."
        ),
        critical=True,
    )

    # 7) Varsity sports >= 14 (critical)
    vs_count = (extraction.varsity_sports_count or "").strip()
    vs_title = "The athletic department sponsors at least 14 NCAA varsity sports."
    vs_claim = f"{inst} sponsors at least 14 NCAA varsity sports."
    if vs_count:
        vs_claim += f" The answer reports: {vs_count}."
    await _add_criterion_seq_with_sources(
        evaluator,
        must_group,
        "Varsity_Sports_Gte_14",
        vs_title,
        extraction.varsity_sports_urls,
        claim=vs_claim,
        add_ins="Verify the official count of NCAA varsity sports; if a range or 'at least' is shown, ensure it's >= 14.",
        critical=True,
    )

    # 8) Significant stadium renovation/capital projects (critical)
    sp_desc = (extraction.stadium_projects_desc or "").strip()
    sp_title = "Significant stadium renovation or capital improvement projects are planned/underway (as of March 2026)."
    sp_claim = f"As of March 2026, {inst}'s athletic department is planning or undertaking significant stadium renovation or capital improvement projects."
    if sp_desc:
        sp_claim += f" The answer describes: {sp_desc}"
    await _add_criterion_seq_with_sources(
        evaluator,
        must_group,
        "Significant_Stadium_Renovation_or_Capital_Projects",
        sp_title,
        extraction.stadium_projects_urls,
        claim=sp_claim,
        add_ins="Look for active or approved plans, RFPs, construction updates, or official announcements current as of March 2026.",
        critical=True,
    )

    # 9) Student fee increase 2024–2026 to support athletics (critical)
    fee_desc = (extraction.student_fee_increase_desc or "").strip()
    fee_title = "Student fee increases (2024–2026) implemented or approved to support athletics."
    fee_claim = f"Between 2024 and 2026, {inst} implemented or approved student fee increases to support athletics."
    if fee_desc:
        fee_claim += f" Details noted in the answer: {fee_desc}"
    await _add_criterion_seq_with_sources(
        evaluator,
        must_group,
        "Student_Fee_Increase_Within_Past_Two_Years_2024_2026",
        fee_title,
        extraction.student_fee_increase_urls,
        claim=fee_claim,
        add_ins="Check student government, university, or official news/board documentation that ties fee increases to athletics (2024–2026).",
        critical=True,
    )

    # 10) State legislative funding in the 2026 session (critical)
    leg_desc = (extraction.state_legislative_funding_desc or "").strip()
    leg_title = "State legislative funding sought/received for athletic facilities/stadium in the 2026 session."
    leg_claim = f"In the 2026 state legislative session, {inst} sought or received state legislative funding for athletic facilities or stadium improvements."
    if leg_desc:
        leg_claim += f" The answer indicates: {leg_desc}"
    await _add_criterion_seq_with_sources(
        evaluator,
        must_group,
        "State_Legislative_Funding_2026_Session",
        leg_title,
        extraction.state_legislative_funding_urls,
        claim=leg_claim,
        add_ins="Look for state legislative bills, budgets, appropriations, or official requests tied to 2026 session funding for athletics/stadium.",
        critical=True,
    )

    # 11) AD oversees NCAA compliance & eligibility (critical)
    await _add_criterion_seq_with_sources(
        evaluator,
        must_group,
        "AD_Oversees_NCAA_Compliance_and_Eligibility",
        "The athletic director is responsible for oversight of NCAA rules compliance and athlete eligibility.",
        extraction.ad_compliance_oversight_urls,
        claim=f"The athletic director at {inst} is responsible for oversight of NCAA rules compliance and athlete eligibility.",
        add_ins="Confirm via job description, organizational chart, or official statements that the AD oversees NCAA compliance and eligibility.",
        critical=True,
    )

    # ------------------------- SHOULD (non-critical) --------------------------
    # NIL program / collective (non-critical)
    nil_desc = (extraction.nil_program_desc or "").strip()
    nil_title = "NIL program or associated collective exists."
    nil_claim = f"{inst} has a Name, Image, and Likeness (NIL) program or operates with an associated NIL collective."
    if nil_desc:
        nil_claim += f" The answer describes: {nil_desc}"
    await _add_criterion_seq_with_sources(
        evaluator,
        should_group,
        "NIL_Program_or_Collective",
        nil_title,
        extraction.nil_program_urls,
        claim=nil_claim,
        add_ins="Verify existence of an NIL program/department or a recognized NIL collective affiliated with the institution/athletics.",
        critical=False,
    )

    # NACDA membership required (non-critical)
    await _add_criterion_seq_with_sources(
        evaluator,
        should_group,
        "NACDA_Membership_Required",
        "The AD position requires membership in NACDA.",
        extraction.nacda_membership_requirement_urls,
        claim=f"The athletic director position at {inst} requires membership in NACDA (National Association of Collegiate Directors of Athletics).",
        add_ins="Look for job postings or HR descriptions that specify NACDA membership as a requirement.",
        critical=False,
    )

    # Advanced degree required or preferred (non-critical)
    await _add_criterion_seq_with_sources(
        evaluator,
        should_group,
        "Advanced_Degree_Required_or_Preferred",
        "The AD position requires or prefers a Master's degree or higher.",
        extraction.advanced_degree_requirement_urls,
        claim=f"The athletic director position at {inst} requires or prefers candidates with a Master's degree or higher.",
        add_ins="Look for job postings/official descriptions stating Master's (or higher) required or preferred.",
        critical=False,
    )

    # Return evaluation summary
    return evaluator.get_summary()