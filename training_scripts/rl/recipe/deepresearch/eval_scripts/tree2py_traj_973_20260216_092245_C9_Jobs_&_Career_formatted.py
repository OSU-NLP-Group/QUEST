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
TASK_ID = "nfl_to_college_transition_nc_2024"
TASK_DESCRIPTION = (
    "In December 2024, a 72-year-old former NFL head coach accepted a college head football coaching position at a "
    "university in North Carolina, signing a five-year contract worth $10 million annually ($1 million base salary plus "
    "$9 million in supplemental income). His inaugural 2025 season ended with a 4-8 record, missing bowl game qualification. "
    "Notably, his son serves as defensive coordinator, while offensive and special teams coordinator positions have also been filled. "
    "This coach is pursuing membership in an elite group of coaches who have won both an NFL Super Bowl and a college football national "
    "championship—currently consisting of only three individuals. Provide comprehensive documentation of this career transition, including: "
    "the coach's full name, the institution's name, the contract expiration date, detailed first-season performance metrics, the defensive "
    "coordinator's full name and his relationship to the head coach, the names of the offensive and special teams coordinators, and the exact "
    "number of coaches who have previously won both championship types. Include supporting reference URLs for each major information category."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class IdentityHiringInfo(BaseModel):
    coach_full_name: Optional[str] = None
    coach_age_at_hiring: Optional[str] = None  # e.g., "72"
    transition_from_role: Optional[str] = None  # e.g., "NFL head coach"
    transition_to_role: Optional[str] = None  # e.g., "college head football coach"
    institution_name: Optional[str] = None
    institution_state: Optional[str] = None  # e.g., "North Carolina"
    hiring_announcement_date: Optional[str] = None  # e.g., "December 2024"
    identity_and_hiring_urls: List[str] = Field(default_factory=list)


class ContractTerms(BaseModel):
    contract_length_years: Optional[str] = None  # e.g., "5 years"
    contract_expiration_date: Optional[str] = None  # e.g., "December 2029"
    total_annual_compensation: Optional[str] = None  # e.g., "$10 million"
    base_salary: Optional[str] = None  # e.g., "$1 million"
    supplemental_income: Optional[str] = None  # e.g., "$9 million"
    contract_urls: List[str] = Field(default_factory=list)


class FirstSeasonPerformance(BaseModel):
    season_year: Optional[str] = None  # e.g., "2025"
    record: Optional[str] = None  # e.g., "4-8" or "4–8"
    bowl_qualification_outcome: Optional[str] = None  # e.g., "missed bowl", "did not qualify"
    performance_urls: List[str] = Field(default_factory=list)


class StaffComposition(BaseModel):
    defensive_coordinator_name: Optional[str] = None
    dc_relationship_to_head_coach: Optional[str] = None  # e.g., "son"
    offensive_coordinator_name: Optional[str] = None
    special_teams_coordinator_name: Optional[str] = None
    staff_urls: List[str] = Field(default_factory=list)


class HistoricalContext(BaseModel):
    elite_group_definition: Optional[str] = None  # e.g., "won both NFL Super Bowl and college national championship"
    coach_attempting_to_join: Optional[str] = None  # e.g., "yes", "attempting"
    number_of_coaches_with_both: Optional[str] = None  # e.g., "3"
    historical_context_urls: List[str] = Field(default_factory=list)


class TransitionDocumentationExtraction(BaseModel):
    identity_hiring: Optional[IdentityHiringInfo] = None
    contract_terms: Optional[ContractTerms] = None
    first_season: Optional[FirstSeasonPerformance] = None
    staff: Optional[StaffComposition] = None
    historical_context: Optional[HistoricalContext] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_transition_doc() -> str:
    return """
    Extract the comprehensive documentation of the specified NFL-to-college head coaching transition from the answer.
    Return a JSON object with the following nested sections and fields. Extract exactly what is explicitly stated in the answer text.
    If a field is missing, set it to null; for URL arrays, return an empty array if none are provided.

    identity_hiring:
      - coach_full_name: The coach's full name.
      - coach_age_at_hiring: The coach's age at the time of hiring (e.g., "72").
      - transition_from_role: The previous role (e.g., "NFL head coach").
      - transition_to_role: The new role (e.g., "college head football coach").
      - institution_name: The university/institution name.
      - institution_state: The state where the institution is located (e.g., "North Carolina").
      - hiring_announcement_date: Month and year of hiring announcement (e.g., "December 2024").
      - identity_and_hiring_urls: All URLs that support the coach identity and hiring/transition announcement details.

    contract_terms:
      - contract_length_years: The stated contract length (e.g., "5 years").
      - contract_expiration_date: The explicit expiration (e.g., "December 2029").
      - total_annual_compensation: The total annual compensation (e.g., "$10 million").
      - base_salary: The base salary amount (e.g., "$1 million").
      - supplemental_income: The supplemental income amount (e.g., "$9 million").
      - contract_urls: All URLs that support contract length/expiration and compensation terms.

    first_season:
      - season_year: The inaugural season year (e.g., "2025").
      - record: The season record (e.g., "4-8" or "4–8").
      - bowl_qualification_outcome: Whether a bowl was qualified/missed (e.g., "missed bowl", "did not qualify").
      - performance_urls: URLs that support the first-season record and bowl outcome.

    staff:
      - defensive_coordinator_name: The defensive coordinator's full name.
      - dc_relationship_to_head_coach: The relationship to the head coach (e.g., "son").
      - offensive_coordinator_name: The offensive coordinator's name.
      - special_teams_coordinator_name: The special teams coordinator's name.
      - staff_urls: URLs that support the coordinator hires/assignments (DC/OC/ST).

    historical_context:
      - elite_group_definition: The definition of the elite group (e.g., winning both an NFL Super Bowl and a college football national championship).
      - coach_attempting_to_join: Whether the coach is attempting to join this elite group (e.g., "yes").
      - number_of_coaches_with_both: The exact number of coaches who have previously won both (e.g., "3").
      - historical_context_urls: URLs that support the historical claim and the count.

    IMPORTANT URL RULES:
    - Extract only URLs explicitly present in the answer (including plain URLs or markdown links).
    - If a URL is missing protocol, prepend http://.
    - If the answer mentions a source without an actual URL, return an empty array for that section's URLs.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_identity_hiring_checks(
    evaluator: Evaluator,
    parent_node,
    info: Optional[IdentityHiringInfo],
):
    node = evaluator.add_parallel(
        id="Identify_Coach_and_Hire_Context",
        desc="Identify the coach and verify the hiring/transition context (who/where/when/what role).",
        parent=parent_node,
        critical=True
    )

    urls = info.identity_and_hiring_urls if info else []

    # Coach full name provided
    evaluator.add_custom_node(
        result=_non_empty(info.coach_full_name) if info else False,
        id="Coach_Full_Name_Provided",
        desc="Answer provides the coach's full name.",
        parent=node,
        critical=True
    )

    # Coach age at hiring is 72
    leaf_age = evaluator.add_leaf(
        id="Coach_Age_At_Hiring_Is_72",
        desc="Coach is stated/verified to be 72 years old at the time of hiring.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="At the time of the December 2024 hiring, the coach was 72 years old.",
        node=leaf_age,
        sources=urls,
        additional_instruction="Confirm the coach's age as 72 in context of the December 2024 announcement; allow reasonable wording like 'age 72'."
    )

    # Transition from NFL head coach to college head coach
    leaf_transition = evaluator.add_leaf(
        id="Transition_From_NFL_Head_Coach_To_College_Head_Coach",
        desc="Coach is verified to have transitioned from an NFL head coaching position to a college head football coaching position.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The coach transitioned from an NFL head coach role to a college head football head coach position.",
        node=leaf_transition,
        sources=urls,
        additional_instruction="The evidence should clearly indicate prior NFL head-coaching and new college head football coach appointment."
    )

    # Institution name provided
    evaluator.add_custom_node(
        result=_non_empty(info.institution_name) if info else False,
        id="Institution_Name_Provided",
        desc="Answer provides the institution/university name.",
        parent=node,
        critical=True
    )

    # Institution located in North Carolina
    leaf_nc = evaluator.add_leaf(
        id="Institution_Located_In_North_Carolina",
        desc="Institution is verified to be located in North Carolina.",
        parent=node,
        critical=True
    )
    inst = info.institution_name if info and info.institution_name else "the institution"
    await evaluator.verify(
        claim=f"{inst} is located in North Carolina.",
        node=leaf_nc,
        sources=urls,
        additional_instruction="Confirm the institution's location explicitly states North Carolina."
    )

    # Hiring announced in December 2024
    leaf_hire_date = evaluator.add_leaf(
        id="Hiring_Announced_In_December_2024",
        desc="Hiring/appointment is verified to have been announced in December 2024.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hiring/appointment was announced in December 2024.",
        node=leaf_hire_date,
        sources=urls,
        additional_instruction="Accept wording like 'announced in December 2024' or a specific December 2024 date."
    )

    # URL presence for identity & hiring
    evaluator.add_custom_node(
        result=_has_urls(urls),
        id="URL_For_Identity_And_Hiring",
        desc="Provides at least one supporting reference URL covering the coach identity + hiring/transition announcement details.",
        parent=node,
        critical=True
    )


async def build_contract_terms_checks(
    evaluator: Evaluator,
    parent_node,
    contract: Optional[ContractTerms],
):
    node = evaluator.add_parallel(
        id="Contract_Terms",
        desc="Verify all required contract terms and provide supporting URL(s).",
        parent=parent_node,
        critical=True
    )
    urls = contract.contract_urls if contract else []

    # URL presence
    evaluator.add_custom_node(
        result=_has_urls(urls),
        id="URL_For_Contract_Terms",
        desc="Provides at least one supporting reference URL for contract length/expiration and compensation terms.",
        parent=node,
        critical=True
    )

    # 5-year deal
    leaf_len = evaluator.add_leaf(
        id="Contract_Is_5_Year_Deal",
        desc="Contract is verified as a 5-year deal.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The coaching contract is a 5-year deal.",
        node=leaf_len,
        sources=urls,
        additional_instruction="Look for explicit mention of a five-year term."
    )

    # Expiration December 2029
    leaf_exp = evaluator.add_leaf(
        id="Contract_Extends_Through_December_2029",
        desc="Contract is verified to extend through / expire in December 2029 (expiration date stated).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The contract extends through December 2029 (expires in December 2029).",
        node=leaf_exp,
        sources=urls,
        additional_instruction="Accept equivalent phrasing like 'through Dec. 2029' or 'expires December 2029'."
    )

    # Total annual compensation $10M
    leaf_total = evaluator.add_leaf(
        id="Annual_Compensation_Total_Is_10M",
        desc="Total annual compensation is verified as $10 million per year.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The coach's total annual compensation is $10 million per year.",
        node=leaf_total,
        sources=urls,
        additional_instruction="Confirm exact figure; allow variants like '$10,000,000 annually'."
    )

    # Base salary $1M
    leaf_base = evaluator.add_leaf(
        id="Annual_Base_Salary_Is_1M",
        desc="Annual base salary is verified as $1 million.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The annual base salary is $1 million.",
        node=leaf_base,
        sources=urls,
        additional_instruction="Confirm explicitly stated base salary amount."
    )

    # Supplemental income $9M
    leaf_supp = evaluator.add_leaf(
        id="Annual_Supplemental_Income_Is_9M",
        desc="Annual supplemental income is verified as $9 million.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The annual supplemental income is $9 million.",
        node=leaf_supp,
        sources=urls,
        additional_instruction="Confirm explicit supplemental/other compensation listed as $9 million."
    )


async def build_first_season_performance_checks(
    evaluator: Evaluator,
    parent_node,
    perf: Optional[FirstSeasonPerformance],
):
    node = evaluator.add_parallel(
        id="First_Season_Performance",
        desc="Verify required first-season timing and outcomes and provide supporting URL(s).",
        parent=parent_node,
        critical=True
    )
    urls = perf.performance_urls if perf else []

    # URL presence
    evaluator.add_custom_node(
        result=_has_urls(urls),
        id="URL_For_First_Season_Performance",
        desc="Provides at least one supporting reference URL for the 2025 season record and bowl-qualification outcome.",
        parent=node,
        critical=True
    )

    # First season is 2025
    leaf_year = evaluator.add_leaf(
        id="First_Season_Is_2025",
        desc="Coach's inaugural college head-coaching season is verified to be 2025.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The coach's inaugural college head-coaching season was 2025.",
        node=leaf_year,
        sources=urls,
        additional_instruction="Confirm that the first season under this coach is 2025."
    )

    # Record is 4–8
    leaf_record = evaluator.add_leaf(
        id="First_Season_Record_Is_4_8",
        desc="First season record is verified as 4 wins and 8 losses (4–8).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The team's 2025 record was 4–8.",
        node=leaf_record,
        sources=urls,
        additional_instruction="Accept '4-8' or '4–8' (hyphen or en dash)."
    )

    # Missed bowl qualification
    leaf_bowl = evaluator.add_leaf(
        id="Missed_Bowl_Qualification_In_First_Season",
        desc="Team is verified to have failed to qualify for a bowl game in that first season.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The team did not qualify for a bowl game in the 2025 season.",
        node=leaf_bowl,
        sources=urls,
        additional_instruction="Evidence should indicate no bowl bid/appearance for 2025."
    )


async def build_staff_composition_checks(
    evaluator: Evaluator,
    parent_node,
    staff: Optional[StaffComposition],
):
    node = evaluator.add_parallel(
        id="Coaching_Staff_Composition",
        desc="Verify coordinator roles and relationships and provide supporting URL(s).",
        parent=parent_node,
        critical=True
    )
    urls = staff.staff_urls if staff else []

    # URL presence
    evaluator.add_custom_node(
        result=_has_urls(urls),
        id="URL_For_Coaching_Staff",
        desc="Provides at least one supporting reference URL for the coordinator hires/assignments (DC/OC/ST).",
        parent=node,
        critical=True
    )

    # DC name provided
    evaluator.add_custom_node(
        result=_non_empty(staff.defensive_coordinator_name) if staff else False,
        id="Defensive_Coordinator_Full_Name_Provided",
        desc="Answer provides the defensive coordinator's full name.",
        parent=node,
        critical=True
    )

    # DC relationship is head coach's son
    leaf_dc_rel = evaluator.add_leaf(
        id="Defensive_Coordinator_Is_Head_Coachs_Son",
        desc="Defensive coordinator is verified to be the head coach's son (relationship stated explicitly).",
        parent=node,
        critical=True
    )
    dc_name = staff.defensive_coordinator_name if staff and staff.defensive_coordinator_name else "The defensive coordinator"
    await evaluator.verify(
        claim=f"{dc_name} is the head coach's son.",
        node=leaf_dc_rel,
        sources=urls,
        additional_instruction="The source should explicitly indicate the familial relationship (son)."
    )

    # OC name provided
    evaluator.add_custom_node(
        result=_non_empty(staff.offensive_coordinator_name) if staff else False,
        id="Offensive_Coordinator_Name_Provided",
        desc="Answer provides the offensive coordinator's name (position filled).",
        parent=node,
        critical=True
    )

    # ST coordinator name provided
    evaluator.add_custom_node(
        result=_non_empty(staff.special_teams_coordinator_name) if staff else False,
        id="Special_Teams_Coordinator_Name_Provided",
        desc="Answer provides the special teams coordinator's name (position filled).",
        parent=node,
        critical=True
    )


async def build_historical_context_checks(
    evaluator: Evaluator,
    parent_node,
    hist: Optional[HistoricalContext],
):
    node = evaluator.add_parallel(
        id="Historical_Championship_Context",
        desc="Verify the historical context claim and provide supporting URL(s).",
        parent=parent_node,
        critical=True
    )
    urls = hist.historical_context_urls if hist else []

    # URL presence
    evaluator.add_custom_node(
        result=_has_urls(urls),
        id="URL_For_Historical_Context",
        desc="Provides at least one supporting reference URL for the historical claim and the count.",
        parent=node,
        critical=True
    )

    # Elite group defined correctly
    leaf_def = evaluator.add_leaf(
        id="Elite_Group_Defined_Correctly",
        desc="Answer states/verifies the elite category is winning both an NFL Super Bowl and a college football national championship.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The elite group is defined as coaches who have won both an NFL Super Bowl and a college football national championship.",
        node=leaf_def,
        sources=urls,
        additional_instruction="Confirm the definition exactly matches winning both titles (NFL Super Bowl + college football national championship)."
    )

    # Coach attempting to join elite group
    leaf_attempt = evaluator.add_leaf(
        id="Coach_Is_Attempting_To_Join_Elite_Group",
        desc="Answer states/verifies that the coach is attempting to join this elite group.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The coach is attempting to join this elite group of coaches who have won both titles.",
        node=leaf_attempt,
        sources=urls,
        additional_instruction="Evidence should clearly refer to pursuing or aiming to join the group."
    )

    # Number of coaches with both is 3
    leaf_count = evaluator.add_leaf(
        id="Number_Of_Coaches_With_Both_Is_3",
        desc="Answer provides the exact number of coaches who have previously won both championship types as 3.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Exactly three coaches have previously won both an NFL Super Bowl and a college football national championship.",
        node=leaf_count,
        sources=urls,
        additional_instruction="Confirm the exact count is three; accept if clearly stated by reputable sources."
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
    Evaluate an answer for the NFL-to-college coaching transition documentation task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Create the top-level critical sequential node (as per rubric)
    doc_root = evaluator.add_sequential(
        id="Career_Transition_Documentation",
        desc="Verify all required facts about the specified NFL-to-college head-coaching transition and provide supporting URLs per major category.",
        parent=root,
        critical=True
    )

    # Extract structured information
    extraction = await evaluator.extract(
        prompt=prompt_extract_transition_doc(),
        template_class=TransitionDocumentationExtraction,
        extraction_name="transition_documentation"
    )

    # Build identity/hiring checks
    await build_identity_hiring_checks(
        evaluator=evaluator,
        parent_node=doc_root,
        info=extraction.identity_hiring
    )

    # Group for contract + performance + staff + history (parallel, all critical)
    verify_group = evaluator.add_parallel(
        id="Verify_Contract_Performance_Staff_And_Historical_Context",
        desc="Verify contract terms, first-season results, staff composition, and the historical-championship context; include URLs per major category.",
        parent=doc_root,
        critical=True
    )

    # Contract terms checks
    await build_contract_terms_checks(
        evaluator=evaluator,
        parent_node=verify_group,
        contract=extraction.contract_terms
    )

    # First season performance checks
    await build_first_season_performance_checks(
        evaluator=evaluator,
        parent_node=verify_group,
        perf=extraction.first_season
    )

    # Coaching staff composition checks
    await build_staff_composition_checks(
        evaluator=evaluator,
        parent_node=verify_group,
        staff=extraction.staff
    )

    # Historical context checks
    await build_historical_context_checks(
        evaluator=evaluator,
        parent_node=verify_group,
        hist=extraction.historical_context
    )

    # Return structured evaluation summary
    return evaluator.get_summary()