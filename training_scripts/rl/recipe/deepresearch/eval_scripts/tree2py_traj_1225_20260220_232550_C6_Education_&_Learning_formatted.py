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
TASK_ID = "ucla_uconn_coach_identification"
TASK_DESCRIPTION = """
Identify the person who meets all of the following criteria:

1. Served as head football coach at the University of California, Los Angeles (UCLA) from 2012 to 2017
2. At UCLA, compiled a 46-30 overall record
3. At UCLA, led the football team to four bowl game appearances
4. In the 2012 season, won the Pac-12 South Division championship at UCLA
5. During the 2012-2017 period, UCLA was a member of the Pac-12 Conference
6. After leaving UCLA, later served as head football coach at the University of Connecticut from 2021 through November 2025
7. At UConn, led the football team to bowl game appearances in three out of four seasons
8. At UConn, achieved consecutive 9-win seasons (9-4 record in 2024 and 9-3 in the prior season)
9. Left the UConn position in November 2025 to accept a head coaching position at Colorado State University

Provide the person's full name.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UCLASection(BaseModel):
    tenure_start_year: Optional[str] = None
    tenure_end_year: Optional[str] = None
    tenure_sources: List[str] = Field(default_factory=list)

    record: Optional[str] = None  # e.g., "46-30"
    record_sources: List[str] = Field(default_factory=list)

    bowl_appearances_count: Optional[str] = None  # e.g., "four"
    bowl_sources: List[str] = Field(default_factory=list)

    pac12_south_2012_statement: Optional[str] = None  # e.g., "Won Pac-12 South in 2012"
    pac12_south_sources: List[str] = Field(default_factory=list)

    pac12_membership_statement: Optional[str] = None  # e.g., "UCLA was Pac-12 member 2012-2017"
    pac12_membership_sources: List[str] = Field(default_factory=list)


class UConnSection(BaseModel):
    tenure_start_year: Optional[str] = None  # e.g., "2021"
    tenure_end_month_year: Optional[str] = None  # e.g., "November 2025"
    tenure_sources: List[str] = Field(default_factory=list)

    bowl_claim_statement: Optional[str] = None  # e.g., "three bowls in four seasons"
    bowl_sources: List[str] = Field(default_factory=list)

    record_2024: Optional[str] = None  # e.g., "9-4"
    prior_season_record: Optional[str] = None  # e.g., "9-3"
    nine_win_sources: List[str] = Field(default_factory=list)


class DepartureSection(BaseModel):
    departure_month_year: Optional[str] = None  # e.g., "November 2025"
    new_position_school: Optional[str] = None  # e.g., "Colorado State University"
    role_title: Optional[str] = None  # e.g., "Head coach"
    departure_sources: List[str] = Field(default_factory=list)


class CSUSection(BaseModel):
    pac12_status_statement: Optional[str] = None  # e.g., "Colorado State is joining the Pac-12"
    pac12_sources: List[str] = Field(default_factory=list)


class PersonExtraction(BaseModel):
    full_name: Optional[str] = None
    ucla: UCLASection = Field(default_factory=UCLASection)
    uconn: UConnSection = Field(default_factory=UConnSection)
    departure: DepartureSection = Field(default_factory=DepartureSection)
    csu: CSUSection = Field(default_factory=CSUSection)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_person_details() -> str:
    return """
    Extract the person's full name and all cited URLs relevant to each of the specified criteria, exactly as presented in the answer.

    Return a JSON object with the following structure:

    {
      "full_name": string | null,

      "ucla": {
        "tenure_start_year": string | null,
        "tenure_end_year": string | null,
        "tenure_sources": string[],

        "record": string | null,
        "record_sources": string[],

        "bowl_appearances_count": string | null,
        "bowl_sources": string[],

        "pac12_south_2012_statement": string | null,
        "pac12_south_sources": string[],

        "pac12_membership_statement": string | null,
        "pac12_membership_sources": string[]
      },

      "uconn": {
        "tenure_start_year": string | null,
        "tenure_end_month_year": string | null,
        "tenure_sources": string[],

        "bowl_claim_statement": string | null,
        "bowl_sources": string[],

        "record_2024": string | null,
        "prior_season_record": string | null,
        "nine_win_sources": string[]
      },

      "departure": {
        "departure_month_year": string | null,
        "new_position_school": string | null,
        "role_title": string | null,
        "departure_sources": string[]
      },

      "csu": {
        "pac12_status_statement": string | null,
        "pac12_sources": string[]
      }
    }

    Rules:
    - Extract only information explicitly present in the answer.
    - For any missing field, set it to null.
    - For each sources array, include every URL cited in the answer for that criterion. Extract actual URLs (including markdown links).
    - Do not invent or infer any information or URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


def name_or_placeholder(name: Optional[str]) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else "UNKNOWN PERSON"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_ucla_section(evaluator: Evaluator, parent_node, data: PersonExtraction) -> None:
    person = name_or_placeholder(data.full_name)

    ucla_node = evaluator.add_parallel(
        id="ucla_coaching_tenure",
        desc="Verify the person's coaching tenure at UCLA from 2012-2017",
        parent=parent_node,
        critical=True
    )

    # Tenure period
    tenure_node = evaluator.add_parallel(
        id="ucla_tenure_period",
        desc="Served as head football coach at UCLA from 2012 to 2017",
        parent=ucla_node,
        critical=True
    )

    # Existence of tenure sources
    evaluator.add_custom_node(
        result=has_urls(data.ucla.tenure_sources),
        id="ucla_tenure_reference",
        desc="Provide URL reference supporting the UCLA tenure dates",
        parent=tenure_node,
        critical=True
    )

    # Verify tenure claim
    tenure_leaf = evaluator.add_leaf(
        id="tenure_dates_2012_2017",
        desc="Verify the coaching tenure spanned 2012 to 2017",
        parent=tenure_node,
        critical=True
    )
    tenure_claim = f"The person named {person} served as head football coach at UCLA from 2012 through 2017."
    await evaluator.verify(
        claim=tenure_claim,
        node=tenure_leaf,
        sources=data.ucla.tenure_sources,
        additional_instruction="Verify that the cited page(s) explicitly state that this person was UCLA head football coach during 2012–2017."
    )

    # Overall record 46-30
    record_node = evaluator.add_parallel(
        id="ucla_overall_record",
        desc="Compiled a 46-30 overall record at UCLA",
        parent=ucla_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_urls(data.ucla.record_sources),
        id="ucla_record_reference",
        desc="Provide URL reference supporting the 46-30 record",
        parent=record_node,
        critical=True
    )

    record_leaf = evaluator.add_leaf(
        id="record_46_30_verification",
        desc="Verify the 46-30 win-loss record",
        parent=record_node,
        critical=True
    )
    record_claim = f"At UCLA, {person} compiled a 46–30 overall record."
    await evaluator.verify(
        claim=record_claim,
        node=record_leaf,
        sources=data.ucla.record_sources,
        additional_instruction="Check that the source explicitly shows a 46–30 overall record for the person's UCLA head coaching tenure."
    )

    # Bowl appearances: four
    bowl_node = evaluator.add_parallel(
        id="ucla_bowl_appearances",
        desc="Led UCLA to four bowl game appearances from 2012-2017",
        parent=ucla_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_urls(data.ucla.bowl_sources),
        id="ucla_bowl_reference",
        desc="Provide URL reference supporting the four bowl appearances",
        parent=bowl_node,
        critical=True
    )

    bowl_leaf = evaluator.add_leaf(
        id="four_bowl_games_ucla",
        desc="Verify four bowl game appearances during UCLA tenure",
        parent=bowl_node,
        critical=True
    )
    bowl_claim = f"During {person}'s UCLA tenure (2012–2017), the team made four bowl game appearances."
    await evaluator.verify(
        claim=bowl_claim,
        node=bowl_leaf,
        sources=data.ucla.bowl_sources,
        additional_instruction="Verify that the source(s) list or state four bowl appearances for UCLA under this head coach between 2012 and 2017."
    )

    # Pac-12 South Division championship in 2012
    champ_node = evaluator.add_parallel(
        id="pac12_south_championship_2012",
        desc="Won the Pac-12 South Division championship in 2012",
        parent=ucla_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_urls(data.ucla.pac12_south_sources),
        id="championship_reference",
        desc="Provide URL reference supporting the 2012 championship",
        parent=champ_node,
        critical=True
    )

    champ_leaf = evaluator.add_leaf(
        id="division_championship_verification",
        desc="Verify the Pac-12 South Division championship in 2012",
        parent=champ_node,
        critical=True
    )
    champ_claim = f"In 2012, {person}'s UCLA team won the Pac-12 South Division championship."
    await evaluator.verify(
        claim=champ_claim,
        node=champ_leaf,
        sources=data.ucla.pac12_south_sources,
        additional_instruction="Confirm the source explicitly states UCLA won the Pac-12 South Division in the 2012 season under this head coach."
    )

    # UCLA Pac-12 membership during 2012-2017
    membership_node = evaluator.add_parallel(
        id="ucla_pac12_membership",
        desc="UCLA was a member of the Pac-12 Conference during the 2012-2017 period",
        parent=ucla_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_urls(data.ucla.pac12_membership_sources),
        id="pac12_membership_reference",
        desc="Provide URL reference confirming UCLA's Pac-12 membership",
        parent=membership_node,
        critical=True
    )

    membership_leaf = evaluator.add_leaf(
        id="pac12_membership_verification",
        desc="Verify UCLA's Pac-12 membership from 2012-2017",
        parent=membership_node,
        critical=True
    )
    membership_claim = "From 2012 through 2017, UCLA was a member of the Pac-12 Conference."
    await evaluator.verify(
        claim=membership_claim,
        node=membership_leaf,
        sources=data.ucla.pac12_membership_sources,
        additional_instruction="Verify that the source confirms UCLA's conference affiliation as Pac-12 for the 2012–2017 period."
    )


async def build_uconn_section(evaluator: Evaluator, parent_node, data: PersonExtraction) -> None:
    person = name_or_placeholder(data.full_name)

    uconn_node = evaluator.add_parallel(
        id="uconn_coaching_tenure",
        desc="Verify the person's coaching tenure at UConn from 2021-November 2025",
        parent=parent_node,
        critical=True
    )

    # UConn tenure dates
    tenure_node = evaluator.add_parallel(
        id="uconn_tenure_period",
        desc="Served as head football coach at UConn from 2021 through November 2025",
        parent=uconn_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_urls(data.uconn.tenure_sources),
        id="uconn_tenure_reference",
        desc="Provide URL reference supporting UConn tenure dates",
        parent=tenure_node,
        critical=True
    )

    tenure_leaf = evaluator.add_leaf(
        id="uconn_dates_verification",
        desc="Verify coaching tenure at UConn from 2021 through November 2025",
        parent=tenure_node,
        critical=True
    )
    tenure_claim = f"The person named {person} served as head football coach at the University of Connecticut from 2021 through November 2025."
    await evaluator.verify(
        claim=tenure_claim,
        node=tenure_leaf,
        sources=data.uconn.tenure_sources,
        additional_instruction="Verify that the source explicitly indicates this person was UConn head football coach starting in 2021 and still in the role through November 2025."
    )

    # UConn bowl appearances claim
    bowls_node = evaluator.add_parallel(
        id="uconn_bowl_appearances",
        desc="Led UConn to bowl game appearances in three out of four seasons",
        parent=uconn_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_urls(data.uconn.bowl_sources),
        id="uconn_bowl_reference",
        desc="Provide URL reference supporting the bowl appearances",
        parent=bowls_node,
        critical=True
    )

    bowls_leaf = evaluator.add_leaf(
        id="three_of_four_bowls",
        desc="Verify three bowl appearances in four seasons at UConn",
        parent=bowls_node,
        critical=True
    )
    bowls_claim = f"Under {person} at UConn, the football team appeared in bowl games in three out of four seasons."
    await evaluator.verify(
        claim=bowls_claim,
        node=bowls_leaf,
        sources=data.uconn.bowl_sources,
        additional_instruction="Confirm that the cited sources enumerate three bowl appearances within a four-season span while this person was UConn head coach."
    )

    # Consecutive 9-win seasons
    nine_node = evaluator.add_parallel(
        id="uconn_consecutive_nine_win_seasons",
        desc="Achieved consecutive 9-win seasons with 9-4 in 2024 and 9-3 in prior season",
        parent=uconn_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_urls(data.uconn.nine_win_sources),
        id="nine_win_reference",
        desc="Provide URL reference supporting the consecutive 9-win seasons",
        parent=nine_node,
        critical=True
    )

    nine_leaf = evaluator.add_leaf(
        id="nine_win_seasons",
        desc="Verify 9-4 record in 2024 and 9-3 record in prior season",
        parent=nine_node,
        critical=True
    )
    nine_claim = f"At UConn, {person} achieved consecutive 9-win seasons: a 9–4 record in 2024 and a 9–3 record in the prior season."
    await evaluator.verify(
        claim=nine_claim,
        node=nine_leaf,
        sources=data.uconn.nine_win_sources,
        additional_instruction="Verify the records and the consecutive nature using the cited sources (season summaries, official athletics pages, or reputable reports)."
    )


async def build_departure_section(evaluator: Evaluator, parent_node, data: PersonExtraction) -> None:
    person = name_or_placeholder(data.full_name)

    depart_node = evaluator.add_parallel(
        id="departure_to_colorado_state",
        desc="Verify the person's departure from UConn to Colorado State",
        parent=parent_node,
        critical=True
    )

    nov_node = evaluator.add_parallel(
        id="november_2025_departure",
        desc="Left UConn in November 2025 to accept head coaching position at Colorado State",
        parent=depart_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_urls(data.departure.departure_sources),
        id="departure_reference",
        desc="Provide URL reference supporting the departure",
        parent=nov_node,
        critical=True
    )

    # Departure timing
    timing_leaf = evaluator.add_leaf(
        id="departure_timing_november_2025",
        desc="Verify departure occurred in November 2025",
        parent=nov_node,
        critical=True
    )
    timing_claim = f"{person} left the UConn head coaching position in November 2025."
    await evaluator.verify(
        claim=timing_claim,
        node=timing_leaf,
        sources=data.departure.departure_sources,
        additional_instruction="Verify that the source gives November 2025 as the departure date from UConn for this person."
    )

    # New CSU position
    csu_pos_leaf = evaluator.add_leaf(
        id="colorado_state_position",
        desc="Verify new position is head coach at Colorado State University",
        parent=nov_node,
        critical=True
    )
    csu_pos_claim = f"After leaving UConn, {person} accepted the head football coaching position at Colorado State University."
    await evaluator.verify(
        claim=csu_pos_claim,
        node=csu_pos_leaf,
        sources=data.departure.departure_sources,
        additional_instruction="Verify that the source explicitly states the new role is 'head coach' at Colorado State University."
    )


async def build_csu_optional_section(evaluator: Evaluator, root_node, data: PersonExtraction) -> None:
    # Optional, non-critical section about CSU and Pac-12 status
    csu_node = evaluator.add_parallel(
        id="colorado_state_pac12_membership",
        desc="Colorado State is joining the Pac-12 Conference",
        parent=root_node,  # Place under non-critical root layer to satisfy critical-child constraint
        critical=False
    )

    evaluator.add_custom_node(
        result=has_urls(data.csu.pac12_sources),
        id="csu_pac12_reference",
        desc="Provide URL reference supporting Colorado State's Pac-12 status",
        parent=csu_node,
        critical=False
    )

    csu_leaf = evaluator.add_leaf(
        id="csu_pac12_verification",
        desc="Verify Colorado State's membership or planned membership in Pac-12",
        parent=csu_node,
        critical=False
    )
    csu_claim = "Colorado State University is joining or planning to join the Pac-12 Conference."
    await evaluator.verify(
        claim=csu_claim,
        node=csu_leaf,
        sources=data.csu.pac12_sources,
        additional_instruction="Check if the cited page(s) confirm CSU's membership or planned admission to the Pac-12 Conference."
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
    Evaluate an answer for the coaching identification task using the Mind2Web2 evaluation framework.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator
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
        prompt=prompt_extract_person_details(),
        template_class=PersonExtraction,
        extraction_name="person_and_sources"
    )

    # Build a critical task main node under root to enforce mandatory criteria
    task_main = evaluator.add_parallel(
        id="task_main",
        desc="Identify the person who meets all specified coaching career and conference-related criteria",
        parent=root,
        critical=True
    )

    # Critical: full name must be provided
    evaluator.add_custom_node(
        result=bool(extracted.full_name and extracted.full_name.strip()),
        id="person_full_name",
        desc="Provide the person's full name",
        parent=task_main,
        critical=True
    )

    # Build sections under the critical task_main
    await build_ucla_section(evaluator, task_main, extracted)
    await build_uconn_section(evaluator, task_main, extracted)
    await build_departure_section(evaluator, task_main, extracted)

    # Optional non-critical CSU Pac-12 status section (placed directly under root to satisfy critical-child constraint)
    await build_csu_optional_section(evaluator, root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()