import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "g5_coaches_eval"
TASK_DESCRIPTION = """I am conducting research on career progression pathways for college football head coaches at the Group of Five level. I need to identify three current FBS head coaches who exemplify successful career development and meet specific professional criteria.

Please find three head coaches who satisfy ALL of the following requirements:

1. Currently serving as the head coach (not interim) at an FBS Group of Five conference school (American Athletic Conference, Conference USA, Mid-American Conference, Mountain West Conference, or Sun Belt Conference)

2. Have at least 15 years of total coaching experience in college football

3. Have at least 3 years of head coaching experience at either the FBS or FCS level

4. Have previous experience as either an offensive coordinator or defensive coordinator at the college level

5. Have coaching experience at at least 2 different institutions

6. Have an overall winning record as a head coach (winning percentage greater than .500)

7. Have held their current head coaching position for at least 2 full completed seasons (as of the end of the 2025 season)

8. Have an active contract extending through at least the 2026 season

9. Have at least one conference championship appearance or bowl game appearance as a head coach

10. Have a profile available on their current university's official athletics website

For each of the three coaches, please provide:
- The coach's full name
- Current university and position
- Total years of coaching experience
- Years of head coaching experience
- Previous coordinator position held
- Overall head coaching record and winning percentage
- Number of completed seasons at current institution
- Contract end year
- Conference championship appearances and/or bowl game appearances
- URL to the coach's official athletics profile
- URL to a source verifying their contract information or career history
"""

REFERENCE_SEASON = 2025
MIN_CONTRACT_THROUGH_YEAR = 2026
GROUP_OF_FIVE_CONFERENCES = [
    "American Athletic Conference", "AAC",
    "Conference USA", "C-USA",
    "Mid-American Conference", "MAC",
    "Mountain West Conference", "MWC",
    "Sun Belt Conference", "Sun Belt"
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CoachInfo(BaseModel):
    name: Optional[str] = None
    current_university: Optional[str] = None
    current_position: Optional[str] = None
    current_conference: Optional[str] = None

    total_coaching_years: Optional[str] = None
    head_coaching_years: Optional[str] = None
    previous_coordinator_position: Optional[str] = None

    overall_head_coaching_record: Optional[str] = None
    winning_percentage: Optional[str] = None

    completed_seasons_current_institution: Optional[str] = None
    contract_end_year: Optional[str] = None

    postseason_appearances: Optional[str] = None  # Text summary ok

    official_profile_url: Optional[str] = None
    contract_or_career_url: Optional[str] = None
    additional_source_urls: List[str] = Field(default_factory=list)

    coaching_institutions: List[str] = Field(default_factory=list)


class CoachesExtraction(BaseModel):
    coaches: List[CoachInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coaches() -> str:
    return """
    Extract up to three (3) current FBS Group of Five head coaches described in the answer, capturing the exact fields below for each coach. If more than three coaches are listed, extract only the first three. If fewer than three are present, still output three objects, using nulls or empty arrays for any missing fields.

    For each coach, extract:
    - name: Full name
    - current_university: Current university name
    - current_position: Current title/role (e.g., "Head Coach")
    - current_conference: The conference of the current program if mentioned (e.g., "AAC", "Conference USA")
    - total_coaching_years: Total years of college football coaching experience, exactly as stated (string)
    - head_coaching_years: Years of head coaching experience (FBS or FCS), exactly as stated (string)
    - previous_coordinator_position: The coordinator role previously held (OC or DC), including school if mentioned (string)
    - overall_head_coaching_record: Overall head coach record (e.g., "45–30")
    - winning_percentage: Winning percentage stated (e.g., "0.615" or "61.5%")
    - completed_seasons_current_institution: Number of completed seasons at current school (as of end of 2025), as stated
    - contract_end_year: Contract end year as stated (string)
    - postseason_appearances: Text summary of conference title game or bowl appearances (string)
    - official_profile_url: URL to the coach’s profile on the current university’s official athletics site (explicit URL required)
    - contract_or_career_url: URL to a source verifying contract information or career history (explicit URL required)
    - additional_source_urls: Any other URLs cited for this coach (array; can be empty)
    - coaching_institutions: List of school names where the coach has worked (array; can be empty)

    Rules:
    - Return a JSON object { "coaches": [...] } with exactly three objects.
    - Only include URLs explicitly present in the answer (plain or markdown).
    - If a field isn’t mentioned, set it to null or an empty array accordingly.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def compose_sources(coach: CoachInfo) -> List[str]:
    sources = []
    if coach.official_profile_url and coach.official_profile_url.strip():
        sources.append(coach.official_profile_url.strip())
    if coach.contract_or_career_url and coach.contract_or_career_url.strip():
        sources.append(coach.contract_or_career_url.strip())
    for u in coach.additional_source_urls or []:
        if isinstance(u, str) and u.strip():
            sources.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def safe_str(x: Optional[str]) -> str:
    return x or ""


# --------------------------------------------------------------------------- #
# Verification logic per coach                                                #
# --------------------------------------------------------------------------- #
async def verify_coach(evaluator: Evaluator, parent_node, coach: CoachInfo, coach_index: int) -> None:
    coach_num = coach_index + 1
    coach_node = evaluator.add_parallel(
        id=f"Coach_{coach_num}",
        desc=f"{['First','Second','Third'][coach_index]} identified coach meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # Identification
    ident_node = evaluator.add_parallel(
        id=f"Coach_{coach_num}_Identification",
        desc=f"Basic identification information for the {['first','second','third'][coach_index]} coach",
        parent=coach_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(coach.name and coach.name.strip()),
        id=f"Coach_{coach_num}_Name",
        desc="The coach's full name is provided",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(coach.current_university and coach.current_university.strip() and coach.current_position and coach.current_position.strip()),
        id=f"Coach_{coach_num}_University",
        desc="The current university and position are provided",
        parent=ident_node,
        critical=True
    )

    # Experience Requirements
    exp_node = evaluator.add_parallel(
        id=f"Experience_Requirements_Coach_{coach_num}",
        desc=f"Verify all coaching experience requirements for the {['first','second','third'][coach_index]} coach",
        parent=coach_node,
        critical=True
    )

    # Total coaching years >= 15
    leaf_total_years = evaluator.add_leaf(
        id=f"Total_Coaching_Years_Coach_{coach_num}",
        desc=f"The coach has at least 15 years of total coaching experience in college football",
        parent=exp_node,
        critical=True
    )
    claim_total_years = (
        f"{safe_str(coach.name)} has at least 15 years of college football coaching experience. "
        f"The extracted total years: '{safe_str(coach.total_coaching_years)}'."
    )
    await evaluator.verify(
        claim=claim_total_years,
        node=leaf_total_years,
        sources=compose_sources(coach),
        additional_instruction="Confirm from the provided sources that total college coaching experience is 15+ years. Minor rounding or phrasing differences are acceptable."
    )

    # Head coaching years >= 3 (FBS/FCS)
    leaf_head_years = evaluator.add_leaf(
        id=f"Head_Coaching_Years_Coach_{coach_num}",
        desc="The coach has at least 3 years of head coaching experience at the FBS or FCS level",
        parent=exp_node,
        critical=True
    )
    claim_head_years = (
        f"{safe_str(coach.name)} has at least 3 years of head coaching experience at the FBS or FCS level. "
        f"Extracted head-coaching years: '{safe_str(coach.head_coaching_years)}'."
    )
    await evaluator.verify(
        claim=claim_head_years,
        node=leaf_head_years,
        sources=compose_sources(coach),
        additional_instruction="Use the sources to confirm cumulative head-coaching duration (FBS or FCS) is 3+ seasons."
    )

    # Coordinator experience (OC/DC)
    leaf_coord = evaluator.add_leaf(
        id=f"Coordinator_Experience_Coach_{coach_num}",
        desc="The coach has previous experience as either an offensive coordinator or defensive coordinator at the college level",
        parent=exp_node,
        critical=True
    )
    claim_coord = (
        f"{safe_str(coach.name)} previously served as a college-level coordinator (OC or DC): '{safe_str(coach.previous_coordinator_position)}'."
    )
    await evaluator.verify(
        claim=claim_coord,
        node=leaf_coord,
        sources=compose_sources(coach),
        additional_instruction="Confirm the person held either an offensive coordinator or defensive coordinator role at the college level (school/year may be listed)."
    )

    # Multi-institution experience (>=2 institutions)
    leaf_multi_inst = evaluator.add_leaf(
        id=f"Multi_Institution_Experience_Coach_{coach_num}",
        desc="The coach has coaching experience at at least 2 different institutions",
        parent=exp_node,
        critical=True
    )
    inst_list = coach.coaching_institutions or []
    claim_multi_inst = (
        f"{safe_str(coach.name)} has coached at at least two institutions. "
        f"Institutions listed: {inst_list}."
    )
    await evaluator.verify(
        claim=claim_multi_inst,
        node=leaf_multi_inst,
        sources=compose_sources(coach),
        additional_instruction="Confirm from career history that the person worked at 2+ distinct universities/programs."
    )

    # Current Position Status
    position_node = evaluator.add_parallel(
        id=f"Current_Position_Status_Coach_{coach_num}",
        desc=f"Verify current position requirements for the {['first','second','third'][coach_index]} coach",
        parent=coach_node,
        critical=True
    )

    # Group of Five school
    leaf_g5 = evaluator.add_leaf(
        id=f"Group_of_Five_School_Coach_{coach_num}",
        desc="The coach currently serves as head coach at an FBS Group of Five conference program (AAC, C-USA, MAC, Mountain West, or Sun Belt)",
        parent=position_node,
        critical=True
    )
    conf_text = safe_str(coach.current_conference)
    claim_g5 = (
        f"{safe_str(coach.name)} is the current head coach at {safe_str(coach.current_university)}, "
        f"which competes in {conf_text}. This conference is one of the FBS Group of Five."
    )
    await evaluator.verify(
        claim=claim_g5,
        node=leaf_g5,
        sources=compose_sources(coach),
        additional_instruction="Verify that the current program competes in AAC, Conference USA, MAC, Mountain West, or Sun Belt (allow known abbreviations)."
    )

    # Currently serving as head coach (not interim)
    leaf_current_role = evaluator.add_leaf(
        id=f"Current_Head_Coach_Role_Coach_{coach_num}",
        desc="The coach is currently serving in the head coach position (not interim, not former)",
        parent=position_node,
        critical=True
    )
    claim_current_role = (
        f"As of the end of {REFERENCE_SEASON}, {safe_str(coach.name)} is serving as the head coach "
        f"at {safe_str(coach.current_university)} (not interim). Extracted title: '{safe_str(coach.current_position)}'."
    )
    await evaluator.verify(
        claim=claim_current_role,
        node=leaf_current_role,
        sources=compose_sources(coach),
        additional_instruction="Confirm the page identifies the person as the current head coach (not interim). Consider bios/news that clearly indicate current status."
    )

    # Contract through >= 2026
    leaf_contract = evaluator.add_leaf(
        id=f"Contract_Through_{MIN_CONTRACT_THROUGH_YEAR}_Coach_{coach_num}",
        desc=f"The coach has an active contract extending through at least the {MIN_CONTRACT_THROUGH_YEAR} season",
        parent=position_node,
        critical=True
    )
    claim_contract = (
        f"{safe_str(coach.name)}'s contract end year is '{safe_str(coach.contract_end_year)}', "
        f"which is at least {MIN_CONTRACT_THROUGH_YEAR}."
    )
    await evaluator.verify(
        claim=claim_contract,
        node=leaf_contract,
        sources=compose_sources(coach),
        additional_instruction=f"Use the contract/career source to verify the deal runs through {MIN_CONTRACT_THROUGH_YEAR} or later."
    )

    # Success Record
    success_node = evaluator.add_parallel(
        id=f"Success_Record_Coach_{coach_num}",
        desc=f"Verify success metrics for the {['first','second','third'][coach_index]} coach",
        parent=coach_node,
        critical=True
    )

    # Winning record (> .500)
    leaf_win = evaluator.add_leaf(
        id=f"Winning_Record_Coach_{coach_num}",
        desc="The coach has an overall winning record as a head coach (winning percentage greater than .500)",
        parent=success_node,
        critical=True
    )
    claim_win = (
        f"As a head coach, {safe_str(coach.name)} has an overall winning record (> .500). "
        f"Record: '{safe_str(coach.overall_head_coaching_record)}'; Winning%: '{safe_str(coach.winning_percentage)}'."
    )
    await evaluator.verify(
        claim=claim_win,
        node=leaf_win,
        sources=compose_sources(coach),
        additional_instruction="If a numeric record is provided (e.g., 45–30), infer the win rate. Minor rounding acceptable."
    )

    # Tenure length: completed >= 2 seasons by end of 2025
    leaf_tenure = evaluator.add_leaf(
        id=f"Tenure_Length_Coach_{coach_num}",
        desc=f"The coach has held their current head coaching position for at least 2 full completed seasons as of the end of the {REFERENCE_SEASON} season",
        parent=success_node,
        critical=True
    )
    claim_tenure = (
        f"By end of {REFERENCE_SEASON}, {safe_str(coach.name)} has completed at least 2 full seasons as head coach "
        f"at {safe_str(coach.current_university)}. Extracted completed seasons: '{safe_str(coach.completed_seasons_current_institution)}'."
    )
    await evaluator.verify(
        claim=claim_tenure,
        node=leaf_tenure,
        sources=compose_sources(coach),
        additional_instruction=f"Confirm seasons completed at the current school are >= 2 as of end of {REFERENCE_SEASON}."
    )

    # Postseason achievements (conference title appearance or bowl)
    leaf_postseason = evaluator.add_leaf(
        id=f"Postseason_Achievement_Coach_{coach_num}",
        desc="The coach has at least one conference championship appearance or bowl game appearance as a head coach",
        parent=success_node,
        critical=True
    )
    claim_postseason = (
        f"{safe_str(coach.name)} has at least one conference championship appearance or bowl game appearance as a head coach. "
        f"Details: {safe_str(coach.postseason_appearances)}."
    )
    await evaluator.verify(
        claim=claim_postseason,
        node=leaf_postseason,
        sources=compose_sources(coach),
        additional_instruction="Either a conference title game appearance or any bowl appearance satisfies this requirement."
    )

    # Verification Documentation
    docs_node = evaluator.add_parallel(
        id=f"Verification_Documentation_Coach_{coach_num}",
        desc=f"Verify that required documentation URLs are provided for the {['first','second','third'][coach_index]} coach",
        parent=coach_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(coach.official_profile_url and coach.official_profile_url.strip() and ("http://" in coach.official_profile_url or "https://" in coach.official_profile_url)),
        id=f"Official_Athletics_Profile_URL_Coach_{coach_num}",
        desc="A valid URL to the coach's profile on their current university's official athletics website is provided",
        parent=docs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(coach.contract_or_career_url and coach.contract_or_career_url.strip() and ("http://" in coach.contract_or_career_url or "https://" in coach.contract_or_career_url)),
        id=f"Contract_Career_Verification_URL_Coach_{coach_num}",
        desc="A valid URL to a source verifying the coach's contract information or career history is provided",
        parent=docs_node,
        critical=True
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
    Evaluate an answer for the Group of Five coaches career progression task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Children (three coaches) are independent
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

    # Note: Root must be non-critical to allow partial credit across coaches in this framework.
    # The rubric JSON marked it critical, but obj_task_eval enforces that critical parents must have all critical children.
    # We intentionally set root to non-critical to avoid structural constraints and allow partial results.

    # Add context info about conferences and temporal references
    evaluator.add_custom_info(
        info={
            "reference_season": REFERENCE_SEASON,
            "min_contract_through_year": MIN_CONTRACT_THROUGH_YEAR,
            "group_of_five_conferences": GROUP_OF_FIVE_CONFERENCES
        },
        info_type="context",
        info_name="evaluation_context"
    )

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_coaches(),
        template_class=CoachesExtraction,
        extraction_name="extracted_coaches"
    )

    # Normalize to exactly 3 coaches
    coaches = list(extracted.coaches or [])
    while len(coaches) < 3:
        coaches.append(CoachInfo())
    if len(coaches) > 3:
        coaches = coaches[:3]

    # Build verification subtrees for each coach
    tasks = []
    for idx, coach in enumerate(coaches):
        tasks.append(verify_coach(evaluator, root, coach, idx))
    # Execute verifications sequentially to respect internal dependency logging order
    for t in tasks:
        await t

    # Return structured result
    return evaluator.get_summary()