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
TASK_ID = "harvard_yale_football_2025_compare"
TASK_DESCRIPTION = (
    "I am a high school junior considering playing football at either Harvard or Yale and want to compare both programs "
    "from an educational and athletic perspective. Please provide a comprehensive comparison that includes: "
    "(1) For each school's current (2025 season) head football coach: their name, undergraduate educational background "
    "(institution and graduation year), and how many seasons they have been leading the program. "
    "(2) The official seating capacity of each school's primary football stadium. "
    "(3) At least one type of academic support service available to student-athletes at each institution. "
    "(4) The Ivy League's policy on athletic scholarships and what type of financial aid is available to student-athletes. "
    "(5) The NCAA division and subdivision classification for both schools. "
    "(6) Whether Ivy League football teams are eligible for postseason playoffs as of the 2025 season. "
    "(7) Information about the most recent (2025) Harvard-Yale football game, including which edition of 'The Game' it was and where it was played. "
    "Please include reference URLs for all information provided."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CoachInfo(BaseModel):
    name: Optional[str] = None
    undergrad_institution: Optional[str] = None
    undergrad_grad_year: Optional[str] = None
    seasons_as_head_coach_through_2025: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StadiumInfo(BaseModel):
    name: Optional[str] = None
    official_capacity: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AcademicSupportInfo(BaseModel):
    services: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class SchoolInfo(BaseModel):
    coach: Optional[CoachInfo] = None
    stadium: Optional[StadiumInfo] = None
    academic_support: Optional[AcademicSupportInfo] = None


class NCAAContextInfo(BaseModel):
    ivy_scholarship_policy: Optional[str] = None
    financial_aid_available: Optional[str] = None
    policy_sources: List[str] = Field(default_factory=list)

    ncaa_division_subdivision_harvard: Optional[str] = None
    ncaa_division_subdivision_yale: Optional[str] = None
    ncaa_classification_sources: List[str] = Field(default_factory=list)

    postseason_playoff_eligibility_ivy_2025: Optional[str] = None
    postseason_sources: List[str] = Field(default_factory=list)


class GameInfo(BaseModel):
    edition_of_the_game_2025: Optional[str] = None
    location_2025: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ComparisonExtraction(BaseModel):
    harvard: Optional[SchoolInfo] = None
    yale: Optional[SchoolInfo] = None
    ncaa_context: Optional[NCAAContextInfo] = None
    game_2025: Optional[GameInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_comparison() -> str:
    return """
Extract the following structured information from the answer. Return exactly the fields requested. For every fact category, also collect the reference URLs explicitly provided in the answer (only actual URLs, not just site names).

Structure your JSON as:

{
  "harvard": {
    "coach": {
      "name": string|null,
      "undergrad_institution": string|null,
      "undergrad_grad_year": string|null,
      "seasons_as_head_coach_through_2025": string|null,
      "sources": string[]   // URLs supporting the coach identity, education, and tenure
    },
    "stadium": {
      "name": string|null,
      "official_capacity": string|null,
      "sources": string[]   // URLs supporting stadium capacity
    },
    "academic_support": {
      "services": string[], // at least one service name/description
      "sources": string[]   // URLs supporting the service(s)
    }
  },
  "yale": {
    "coach": {
      "name": string|null,
      "undergrad_institution": string|null,
      "undergrad_grad_year": string|null,
      "seasons_as_head_coach_through_2025": string|null,
      "sources": string[]
    },
    "stadium": {
      "name": string|null,
      "official_capacity": string|null,
      "sources": string[]
    },
    "academic_support": {
      "services": string[],
      "sources": string[]
    }
  },
  "ncaa_context": {
    "ivy_scholarship_policy": string|null,     // e.g., "No athletic scholarships in Ivy League"
    "financial_aid_available": string|null,    // e.g., "Need-based financial aid is available"
    "policy_sources": string[],                // URLs supporting policy/financial aid statements

    "ncaa_division_subdivision_harvard": string|null, // e.g., "NCAA Division I FCS"
    "ncaa_division_subdivision_yale": string|null,    // e.g., "NCAA Division I FCS"
    "ncaa_classification_sources": string[],          // URLs supporting classification for both

    "postseason_playoff_eligibility_ivy_2025": string|null, // e.g., "Not eligible/does not participate in FCS playoffs" or "Eligible"
    "postseason_sources": string[]                          // URLs supporting postseason policy
  },
  "game_2025": {
    "edition_of_the_game_2025": string|null, // e.g., "142nd", "XXXth meeting"
    "location_2025": string|null,            // e.g., "Yale Bowl, New Haven, CT"
    "sources": string[]                      // URLs supporting the 2025 game info
  }
}

Rules:
- Extract only what is explicitly present in the answer.
- All URL fields must contain valid URLs present in the answer; if none, use an empty array.
- Prefer strings for numeric-looking values (e.g., seasons, capacities, years) to maximize robustness.
- If an item is missing in the answer, set it to null (or an empty array for URL lists).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str], default: str = "unknown") -> str:
    return s if (s is not None and str(s).strip() != "") else default


def _first_or_unknown(items: Optional[List[str]]) -> str:
    if items and len(items) > 0 and str(items[0]).strip() != "":
        return items[0]
    return "unknown"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_harvard_section(evaluator: Evaluator, parent_node, data: ComparisonExtraction):
    # Harvard Program Information (parallel, non-critical)
    harvard_node = evaluator.add_parallel(
        id="Harvard_Program_Information",
        desc="Information about Harvard's football program",
        parent=parent_node,
        critical=False,
    )

    harvard = data.harvard or SchoolInfo()

    # Harvard Head Coach Background (parallel, CRITICAL)
    coach_bg_node = evaluator.add_parallel(
        id="Harvard_Head_Coach_Background",
        desc="Background information about Harvard's current (2025 season) head football coach",
        parent=harvard_node,
        critical=True,
    )

    coach = harvard.coach or CoachInfo()

    # Coach Identity (leaf, CRITICAL)
    coach_identity_node = evaluator.add_leaf(
        id="Harvard_Coach_Identity",
        desc="Identifies Harvard's current (2025 season) head football coach by name",
        parent=coach_bg_node,
        critical=True,
    )
    coach_identity_claim = f"As of the 2025 season, the head football coach of Harvard is {_safe(coach.name)}."
    await evaluator.verify(
        claim=coach_identity_claim,
        node=coach_identity_node,
        sources=coach.sources,
        additional_instruction="Confirm the current (2025 season) Harvard head coach name on the provided URL(s)."
    )

    # Coach Education (leaf, CRITICAL)
    coach_edu_node = evaluator.add_leaf(
        id="Harvard_Coach_Education",
        desc="Provides the head coach's undergraduate institution and graduation year",
        parent=coach_bg_node,
        critical=True,
    )
    coach_edu_claim = (
        f"{_safe(coach.name)} completed an undergraduate degree at {_safe(coach.undergrad_institution)} "
        f"in {_safe(coach.undergrad_grad_year)}."
    )
    await evaluator.verify(
        claim=coach_edu_claim,
        node=coach_edu_node,
        sources=coach.sources,
        additional_instruction="Check coach biography or credible sources for undergraduate institution and graduation year."
    )

    # Coach Tenure (leaf, CRITICAL)
    coach_tenure_node = evaluator.add_leaf(
        id="Harvard_Coach_Tenure",
        desc="States how many seasons the coach has been leading the program as of the 2025 season",
        parent=coach_bg_node,
        critical=True,
    )
    coach_tenure_claim = (
        f"As of the 2025 season, {_safe(coach.name)} has been Harvard's head coach for "
        f"{_safe(coach.seasons_as_head_coach_through_2025)} season(s)."
    )
    await evaluator.verify(
        claim=coach_tenure_claim,
        node=coach_tenure_node,
        sources=coach.sources,
        additional_instruction="Verify the number of seasons as head coach through the 2025 season; use hire/start year if needed."
    )

    # Harvard Stadium Capacity (leaf, CRITICAL)
    stadium = harvard.stadium or StadiumInfo()
    stadium_node = evaluator.add_leaf(
        id="Harvard_Stadium_Capacity",
        desc="Provides the official seating capacity of Harvard's primary football stadium",
        parent=harvard_node,
        critical=True,
    )
    stadium_name = _safe(stadium.name, "Harvard's primary football stadium")
    stadium_claim = (
        f"The official seating capacity of {stadium_name} is {_safe(stadium.official_capacity)}."
    )
    await evaluator.verify(
        claim=stadium_claim,
        node=stadium_node,
        sources=stadium.sources,
        additional_instruction="Use official athletics or facility pages to confirm the stadium's official seating capacity."
    )

    # Harvard Academic Support (leaf, CRITICAL)
    acad = harvard.academic_support or AcademicSupportInfo()
    acad_node = evaluator.add_leaf(
        id="Harvard_Academic_Support",
        desc="Identifies at least one academic support service available to Harvard student-athletes",
        parent=harvard_node,
        critical=True,
    )
    service_name = _first_or_unknown(acad.services)
    acad_claim = (
        f"Harvard student-athletes have access to the following academic support service: {service_name}."
    )
    await evaluator.verify(
        claim=acad_claim,
        node=acad_node,
        sources=acad.sources,
        additional_instruction="Confirm that the named service is an academic support offering available to Harvard student-athletes."
    )


async def build_yale_section(evaluator: Evaluator, parent_node, data: ComparisonExtraction):
    # Yale Program Information (parallel, non-critical)
    yale_node = evaluator.add_parallel(
        id="Yale_Program_Information",
        desc="Information about Yale's football program",
        parent=parent_node,
        critical=False,
    )

    yale = data.yale or SchoolInfo()

    # Yale Head Coach Background (parallel, CRITICAL)
    coach_bg_node = evaluator.add_parallel(
        id="Yale_Head_Coach_Background",
        desc="Background information about Yale's current (2025 season) head football coach",
        parent=yale_node,
        critical=True,
    )

    coach = yale.coach or CoachInfo()

    # Coach Identity (leaf, CRITICAL)
    coach_identity_node = evaluator.add_leaf(
        id="Yale_Coach_Identity",
        desc="Identifies Yale's current (2025 season) head football coach by name",
        parent=coach_bg_node,
        critical=True,
    )
    coach_identity_claim = f"As of the 2025 season, the head football coach of Yale is {_safe(coach.name)}."
    await evaluator.verify(
        claim=coach_identity_claim,
        node=coach_identity_node,
        sources=coach.sources,
        additional_instruction="Confirm the current (2025 season) Yale head coach name on the provided URL(s)."
    )

    # Coach Education (leaf, CRITICAL)
    coach_edu_node = evaluator.add_leaf(
        id="Yale_Coach_Education",
        desc="Provides the head coach's undergraduate institution and graduation year",
        parent=coach_bg_node,
        critical=True,
    )
    coach_edu_claim = (
        f"{_safe(coach.name)} completed an undergraduate degree at {_safe(coach.undergrad_institution)} "
        f"in {_safe(coach.undergrad_grad_year)}."
    )
    await evaluator.verify(
        claim=coach_edu_claim,
        node=coach_edu_node,
        sources=coach.sources,
        additional_instruction="Check coach biography or credible sources for undergraduate institution and graduation year."
    )

    # Coach Tenure (leaf, CRITICAL)
    coach_tenure_node = evaluator.add_leaf(
        id="Yale_Coach_Tenure",
        desc="States how many seasons the coach has been leading the program as of the 2025 season",
        parent=coach_bg_node,
        critical=True,
    )
    coach_tenure_claim = (
        f"As of the 2025 season, {_safe(coach.name)} has been Yale's head coach for "
        f"{_safe(coach.seasons_as_head_coach_through_2025)} season(s)."
    )
    await evaluator.verify(
        claim=coach_tenure_claim,
        node=coach_tenure_node,
        sources=coach.sources,
        additional_instruction="Verify the number of seasons as head coach through the 2025 season; use hire/start year if needed."
    )

    # Yale Stadium Capacity (leaf, CRITICAL)
    stadium = yale.stadium or StadiumInfo()
    stadium_node = evaluator.add_leaf(
        id="Yale_Stadium_Capacity",
        desc="Provides the official seating capacity of Yale's primary football stadium",
        parent=yale_node,
        critical=True,
    )
    stadium_name = _safe(stadium.name, "Yale's primary football stadium")
    stadium_claim = (
        f"The official seating capacity of {stadium_name} is {_safe(stadium.official_capacity)}."
    )
    await evaluator.verify(
        claim=stadium_claim,
        node=stadium_node,
        sources=stadium.sources,
        additional_instruction="Use official athletics or facility pages to confirm the stadium's official seating capacity."
    )

    # Yale Academic Support (leaf, CRITICAL)
    acad = yale.academic_support or AcademicSupportInfo()
    acad_node = evaluator.add_leaf(
        id="Yale_Academic_Support",
        desc="Identifies at least one academic support service available to Yale student-athletes",
        parent=yale_node,
        critical=True,
    )
    service_name = _first_or_unknown(acad.services)
    acad_claim = (
        f"Yale student-athletes have access to the following academic support service: {service_name}."
    )
    await evaluator.verify(
        claim=acad_claim,
        node=acad_node,
        sources=acad.sources,
        additional_instruction="Confirm that the named service is an academic support offering available to Yale student-athletes."
    )


async def build_ivy_ncaa_section(evaluator: Evaluator, parent_node, data: ComparisonExtraction):
    # Ivy League and NCAA Context (parallel, non-critical)
    ctx_node = evaluator.add_parallel(
        id="Ivy_League_and_NCAA_Context",
        desc="Shared Ivy League/NCAA context requested in the comparison",
        parent=parent_node,
        critical=False,
    )

    ctx = data.ncaa_context or NCAAContextInfo()

    # Athletic Scholarship and Financial Aid Policy (leaf, CRITICAL)
    policy_leaf = evaluator.add_leaf(
        id="Athletic_Scholarship_and_Financial_Aid_Policy",
        desc="Describes the Ivy League policy on athletic scholarships and what type(s) of financial aid are available to student-athletes",
        parent=ctx_node,
        critical=True,
    )
    # Compose a concise combined policy claim
    policy_text = _safe(ctx.ivy_scholarship_policy)
    aid_text = _safe(ctx.financial_aid_available)
    policy_claim = (
        f"Ivy League policy on athletic scholarships is: {policy_text}. "
        f"The type(s) of financial aid available to student-athletes are: {aid_text}."
    )
    await evaluator.verify(
        claim=policy_claim,
        node=policy_leaf,
        sources=ctx.policy_sources,
        additional_instruction="Verify Ivy League scholarship policy and clarify what financial aid student-athletes can receive (e.g., need-based aid)."
    )

    # NCAA Division and Subdivision (parallel aggregator under context, CRITICAL)
    ncaa_div_node = evaluator.add_parallel(
        id="NCAA_Division_and_Subdivision",
        desc="States the NCAA division and football subdivision classification for both Harvard and Yale",
        parent=ctx_node,
        critical=True,
    )

    # Harvard classification (leaf, CRITICAL)
    harv_div_leaf = evaluator.add_leaf(
        id="NCAA_Division_and_Subdivision_Harvard",
        desc="Harvard NCAA division/subdivision classification",
        parent=ncaa_div_node,
        critical=True,
    )
    harv_div_claim = f"Harvard competes in {_safe(ctx.ncaa_division_subdivision_harvard)} for football."
    await evaluator.verify(
        claim=harv_div_claim,
        node=harv_div_leaf,
        sources=ctx.ncaa_classification_sources,
        additional_instruction="Confirm Harvard's NCAA division and football subdivision (e.g., Division I FCS)."
    )

    # Yale classification (leaf, CRITICAL)
    yale_div_leaf = evaluator.add_leaf(
        id="NCAA_Division_and_Subdivision_Yale",
        desc="Yale NCAA division/subdivision classification",
        parent=ncaa_div_node,
        critical=True,
    )
    yale_div_claim = f"Yale competes in {_safe(ctx.ncaa_division_subdivision_yale)} for football."
    await evaluator.verify(
        claim=yale_div_claim,
        node=yale_div_leaf,
        sources=ctx.ncaa_classification_sources,
        additional_instruction="Confirm Yale's NCAA division and football subdivision (e.g., Division I FCS)."
    )

    # Postseason Playoff Eligibility as of 2025 (leaf, CRITICAL)
    postseason_leaf = evaluator.add_leaf(
        id="Postseason_Playoff_Eligibility_As_of_2025",
        desc="States whether Ivy League football teams are eligible for postseason playoffs as of the 2025 season",
        parent=ctx_node,
        critical=True,
    )
    postseason_text = _safe(ctx.postseason_playoff_eligibility_ivy_2025)
    postseason_claim = f"As of the 2025 season, Ivy League football teams' postseason playoff eligibility is: {postseason_text}."
    await evaluator.verify(
        claim=postseason_claim,
        node=postseason_leaf,
        sources=ctx.postseason_sources,
        additional_instruction="Verify whether Ivy League teams participate in/are eligible for the NCAA FCS playoffs as of 2025."
    )


async def build_game_section(evaluator: Evaluator, parent_node, data: ComparisonExtraction):
    # Most Recent 2025 Harvard–Yale Game (parallel, non-critical)
    game_node = evaluator.add_parallel(
        id="Most_Recent_2025_Harvard_Yale_Game",
        desc="Information about the most recent (2025) Harvard–Yale game",
        parent=parent_node,
        critical=False,
    )

    game = data.game_2025 or GameInfo()

    # Game Edition (leaf, CRITICAL)
    edition_leaf = evaluator.add_leaf(
        id="Game_Edition",
        desc="States which edition of 'The Game' the 2025 contest was",
        parent=game_node,
        critical=True,
    )
    edition_claim = f"The 2025 Harvard–Yale game was the {_safe(game.edition_of_the_game_2025)} edition of 'The Game'."
    await evaluator.verify(
        claim=edition_claim,
        node=edition_leaf,
        sources=game.sources,
        additional_instruction="Confirm the ordinal/edition number (e.g., 142nd) for the 2025 Harvard–Yale matchup."
    )

    # Game Location (leaf, CRITICAL)
    location_leaf = evaluator.add_leaf(
        id="Game_Location",
        desc="States where the 2025 game was played",
        parent=game_node,
        critical=True,
    )
    location_claim = f"The 2025 Harvard–Yale game was played at {_safe(game.location_2025)}."
    await evaluator.verify(
        claim=location_claim,
        node=location_leaf,
        sources=game.sources,
        additional_instruction="Verify the venue (e.g., Yale Bowl or Harvard Stadium) and location for the 2025 game."
    )


async def build_citations_check(evaluator: Evaluator, parent_node, data: ComparisonExtraction):
    """
    Build critical citation presence checks for each requested category.
    Converts the single rubric item 'Citations_and_Reference_URLs' into a critical parallel node
    with individual critical leaf checks to ensure atomic verification steps.
    """
    cite_node = evaluator.add_parallel(
        id="Citations_and_Reference_URLs",
        desc="Includes reference URLs supporting all requested information provided in the response",
        parent=parent_node,
        critical=True,
    )

    harvard = data.harvard or SchoolInfo()
    yale = data.yale or SchoolInfo()
    ctx = data.ncaa_context or NCAAContextInfo()
    game = data.game_2025 or GameInfo()

    # Helper to add a citation existence custom node (critical)
    def add_citation_check(node_id: str, desc: str, urls: Optional[List[str]]):
        result_bool = bool(urls) and len(urls) > 0
        evaluator.add_custom_node(
            result=result_bool,
            id=node_id,
            desc=desc,
            parent=cite_node,
            critical=True
        )

    # Harvard
    add_citation_check("Cite_Harvard_Coach", "Citations present for Harvard head coach facts", (harvard.coach or CoachInfo()).sources)
    add_citation_check("Cite_Harvard_Stadium", "Citations present for Harvard stadium capacity", (harvard.stadium or StadiumInfo()).sources)
    add_citation_check("Cite_Harvard_Academic", "Citations present for Harvard academic support service(s)", (harvard.academic_support or AcademicSupportInfo()).sources)

    # Yale
    add_citation_check("Cite_Yale_Coach", "Citations present for Yale head coach facts", (yale.coach or CoachInfo()).sources)
    add_citation_check("Cite_Yale_Stadium", "Citations present for Yale stadium capacity", (yale.stadium or StadiumInfo()).sources)
    add_citation_check("Cite_Yale_Academic", "Citations present for Yale academic support service(s)", (yale.academic_support or AcademicSupportInfo()).sources)

    # Ivy/NCAA context
    add_citation_check("Cite_Ivy_Policy", "Citations present for Ivy League scholarship/financial aid policy", ctx.policy_sources)
    add_citation_check("Cite_NCAA_Classification", "Citations present for NCAA division/subdivision classifications", ctx.ncaa_classification_sources)
    add_citation_check("Cite_Postseason", "Citations present for postseason eligibility policy as of 2025", ctx.postseason_sources)

    # Game 2025
    add_citation_check("Cite_Game_2025", "Citations present for 2025 Harvard–Yale game edition/location", game.sources)


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
    Evaluate an answer for the Harvard vs. Yale football comparison task (2025 season focus).
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root per rubric: parallel aggregation across sections
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

    # Add a top-level container node matching the rubric root (non-critical, parallel)
    top = evaluator.add_parallel(
        id="Comparison_of_Harvard_and_Yale_Football_Programs",
        desc="Complete comparison of Harvard and Yale football programs covering coaching, facilities, academic support, league/NCAA context, and the most recent Harvard–Yale game, with citations",
        parent=root,
        critical=False,
    )

    # Extract data once
    extraction = await evaluator.extract(
        prompt=prompt_extract_comparison(),
        template_class=ComparisonExtraction,
        extraction_name="comparison_extraction"
    )

    # Build sections according to rubric
    await build_harvard_section(evaluator, top, extraction)
    await build_yale_section(evaluator, top, extraction)
    await build_ivy_ncaa_section(evaluator, top, extraction)
    await build_game_section(evaluator, top, extraction)
    await build_citations_check(evaluator, top, extraction)

    # Return evaluation summary
    return evaluator.get_summary()