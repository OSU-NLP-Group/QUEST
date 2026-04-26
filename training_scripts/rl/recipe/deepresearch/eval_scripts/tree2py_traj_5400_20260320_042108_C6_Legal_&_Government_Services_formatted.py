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
TASK_ID = "immigration_venezuela_2024_2026"
TASK_DESCRIPTION = """
During 2024-2026, significant developments occurred in U.S. immigration enforcement policy, judicial decisions, and international relations with Venezuela. Research and provide the following information:

Part A - Border Patrol Leadership:
Identify the Border Patrol commander who led large-scale immigration enforcement operations in multiple U.S. cities (including Los Angeles, Chicago, New Orleans, and Minneapolis) during 2025-2026. Provide:
1. The commander's full name
2. The commander's birth year
3. The year the commander joined the Border Patrol
4. The month and year when the commander was appointed as commander-at-large
5. The specific Border Patrol sector the commander previously led before this appointment

Part B - Fifth Circuit Court Decision:
In early 2026, a divided panel of the U.S. Court of Appeals for the Fifth Circuit issued a ruling on the Trump administration's mandatory immigration detention policy. Provide:
1. The exact date (month, day, and year) of this decision
2. The name of the judge who wrote the majority opinion
3. Which U.S. President appointed the majority opinion author
4. The name of the judge who dissented
5. The approximate number of federal judges who had ruled against the administration's detention policy before this Fifth Circuit decision

Part C - Supreme Court Immigration Decision (April 2025):
In April 2025, the U.S. Supreme Court issued a decision regarding detention and removal of noncitizens. Provide:
1. The exact date (month, day, and year) of this decision
2. The case name
3. The vote split (e.g., 5-4, 6-3, etc.)
4. Whether Justice Amy Coney Barrett was in the majority or dissent

Part D - U.S.-Venezuela Relations Timeline:
Between late 2025 and early 2026, several major events occurred regarding U.S.-Venezuela relations. Provide:
1. The exact date (month, day, and year) when U.S. forces carried out military strikes on Caracas, Venezuela
2. Confirmation that Nicolás Maduro was captured during this operation
3. The name of the person who became Venezuela's acting or interim president after Maduro's capture
4. The exact date (month, day, and year) when the U.S. and Venezuela agreed to reestablish diplomatic relations
5. The exact date (month, day, and year) when President Trump ordered a naval blockade of Venezuelan oil tankers

For each piece of information, provide a reference URL from a reputable source that confirms the stated fact.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class BorderPatrolCommanderExtraction(BaseModel):
    name: Optional[str] = None
    name_sources: List[str] = Field(default_factory=list)
    birth_year: Optional[str] = None
    birth_year_sources: List[str] = Field(default_factory=list)
    joined_year: Optional[str] = None
    joined_year_sources: List[str] = Field(default_factory=list)
    appointment_month_year: Optional[str] = None
    appointment_sources: List[str] = Field(default_factory=list)
    prior_sector: Optional[str] = None
    prior_sector_sources: List[str] = Field(default_factory=list)


class FifthCircuitExtraction(BaseModel):
    decision_date: Optional[str] = None
    decision_date_sources: List[str] = Field(default_factory=list)
    majority_author: Optional[str] = None
    majority_author_sources: List[str] = Field(default_factory=list)
    appointing_president: Optional[str] = None
    appointing_president_sources: List[str] = Field(default_factory=list)
    dissenting_judge: Optional[str] = None
    dissenting_judge_sources: List[str] = Field(default_factory=list)
    prior_opposition_count: Optional[str] = None
    prior_opposition_count_sources: List[str] = Field(default_factory=list)


class SupremeCourt2025Extraction(BaseModel):
    decision_date: Optional[str] = None
    decision_date_sources: List[str] = Field(default_factory=list)
    case_name: Optional[str] = None
    case_name_sources: List[str] = Field(default_factory=list)
    vote_split: Optional[str] = None
    vote_split_sources: List[str] = Field(default_factory=list)
    barrett_position: Optional[str] = None  # expected values like "majority" or "dissent"
    barrett_position_sources: List[str] = Field(default_factory=list)


class VenezuelaTimelineExtraction(BaseModel):
    strike_date: Optional[str] = None
    strike_date_sources: List[str] = Field(default_factory=list)
    maduro_captured: Optional[str] = None  # "yes"/"no"/"true"/"false" or a statement
    maduro_captured_sources: List[str] = Field(default_factory=list)
    interim_leader: Optional[str] = None
    interim_leader_sources: List[str] = Field(default_factory=list)
    diplomatic_restoration_date: Optional[str] = None
    diplomatic_restoration_date_sources: List[str] = Field(default_factory=list)
    blockade_order_date: Optional[str] = None
    blockade_order_date_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_border_patrol_commander() -> str:
    return """
    Extract information (as provided in the answer) about the Border Patrol commander who led large-scale immigration enforcement operations in multiple U.S. cities (including Los Angeles, Chicago, New Orleans, and Minneapolis) during 2025-2026.

    For each field, also extract the specific reference URL(s) the answer provides for that fact (if any). Only include full, valid URLs explicitly present in the answer text. If missing, return an empty list.

    Required fields:
    - name: Full name of the commander
    - name_sources: URL(s) that confirm the name
    - birth_year: The commander's birth year
    - birth_year_sources: URL(s) confirming the birth year
    - joined_year: The year the commander joined the U.S. Border Patrol
    - joined_year_sources: URL(s) confirming the joined year
    - appointment_month_year: The month and year when appointed as commander-at-large
    - appointment_sources: URL(s) confirming the appointment month/year and role (commander-at-large)
    - prior_sector: The specific Border Patrol sector the commander led before this appointment
    - prior_sector_sources: URL(s) confirming the prior sector leadership

    Rules:
    - Do not fabricate information or URLs; extract exactly what appears in the answer.
    - If a field is not present, set it to null (for strings) or [] (for URL lists).
    """


def prompt_extract_fifth_circuit() -> str:
    return """
    Extract information (as provided in the answer) about the Fifth Circuit decision in early 2026 concerning the Trump administration's mandatory immigration detention policy.

    Required fields, each with its own sources list:
    - decision_date: Exact date (e.g., "February 3, 2026")
    - decision_date_sources: URL(s) confirming the date
    - majority_author: Name of the judge who wrote the majority opinion
    - majority_author_sources: URL(s) confirming the majority author
    - appointing_president: Which U.S. President appointed the majority author
    - appointing_president_sources: URL(s) confirming the appointing president
    - dissenting_judge: Name of the judge who dissented
    - dissenting_judge_sources: URL(s) confirming the dissent
    - prior_opposition_count: Approximate number of federal judges who had ruled against the administration's detention policy before this decision (as stated in the answer)
    - prior_opposition_count_sources: URL(s) supporting that approximate count

    Rules:
    - Extract exactly as in the answer.
    - Use only explicit URLs from the answer.
    """


def prompt_extract_scotus_2025() -> str:
    return """
    Extract information (as provided in the answer) about the April 2025 U.S. Supreme Court decision regarding detention and removal of noncitizens.

    Required fields:
    - decision_date: Exact date (e.g., "April 23, 2025")
    - decision_date_sources: URL(s) confirming the date
    - case_name: Full case name (e.g., "Garland v. X")
    - case_name_sources: URL(s) confirming the case name
    - vote_split: Vote split (e.g., "6-3", "5-4")
    - vote_split_sources: URL(s) confirming the vote split
    - barrett_position: Whether Justice Amy Coney Barrett was in the "majority" or "dissent" (use one of these words)
    - barrett_position_sources: URL(s) confirming Barrett's position

    Rules:
    - Extract exactly as in the answer.
    - Use only explicit URLs from the answer.
    """


def prompt_extract_venezuela() -> str:
    return """
    Extract information (as provided in the answer) about U.S.-Venezuela relations between late 2025 and early 2026.

    Required fields:
    - strike_date: Exact date when U.S. forces carried out military strikes on Caracas
    - strike_date_sources: URL(s) confirming the strike date
    - maduro_captured: A short yes/no-like value or a sentence indicating whether Nicolás Maduro was captured during the operation
    - maduro_captured_sources: URL(s) confirming whether Maduro was captured
    - interim_leader: Name of the person who became Venezuela's acting or interim president after Maduro's capture
    - interim_leader_sources: URL(s) confirming the interim leader
    - diplomatic_restoration_date: Exact date when the U.S. and Venezuela agreed to reestablish diplomatic relations
    - diplomatic_restoration_date_sources: URL(s) confirming that date
    - blockade_order_date: Exact date when President Trump ordered a naval blockade of Venezuelan oil tankers
    - blockade_order_date_sources: URL(s) confirming the blockade order date

    Rules:
    - Extract exactly as in the answer.
    - Use only explicit URLs from the answer.
    - If any field is not present, return null (for string) or [] (for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _clean_sources(urls: Optional[List[str]]) -> List[str]:
    return [u.strip() for u in (urls or []) if isinstance(u, str) and u.strip()]


async def _add_and_verify_with_sources(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent,
    claim: str,
    sources: Optional[List[str]],
    critical: bool = True,
    additional_instruction: Optional[str] = None,
) -> bool:
    """
    Create a leaf node and verify a claim against provided sources.
    If sources are missing/empty, directly mark as failed (to enforce source-grounding).
    """
    node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    srcs = _clean_sources(sources)
    if len(srcs) == 0:
        node.score = 0.0
        node.status = "failed"
        return False

    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=srcs,
        additional_instruction=additional_instruction or (
            "Judge strictly based on the cited page(s). "
            "If the page(s) do not explicitly support the claim, return Not Supported."
        )
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_border_patrol_commander_tree(
    evaluator: Evaluator,
    parent,
    data: BorderPatrolCommanderExtraction
) -> None:
    bp_node = evaluator.add_parallel(
        id="border_patrol_commander",
        desc="Identify the Border Patrol commander who led large-scale immigration enforcement operations in multiple U.S. cities during 2025-2026",
        parent=parent,
        critical=False
    )

    # commander_identity
    ident_group = evaluator.add_parallel(
        id="commander_identity",
        desc="Provide the commander's full name with supporting reference",
        parent=bp_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.name),
        id="factual_accuracy_name",
        desc="The commander's full name is correctly identified (non-empty)",
        parent=ident_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_name",
        desc="Valid reference URL is provided confirming the commander's name",
        parent=ident_group,
        claim=f"The Border Patrol commander who led large-scale immigration enforcement operations in 2025-2026 is {data.name or '[name missing]'}.",
        sources=data.name_sources,
        critical=True,
        additional_instruction="Confirm the individual's name and that they are identified as the relevant Border Patrol commander."
    )

    # commander_birth_year
    birth_group = evaluator.add_parallel(
        id="commander_birth_year",
        desc="Provide the commander's birth year with supporting reference",
        parent=bp_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.birth_year),
        id="factual_accuracy_birth_year",
        desc="The commander's birth year is correctly provided (non-empty)",
        parent=birth_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_birth_year",
        desc="Valid reference URL is provided confirming the birth year",
        parent=birth_group,
        claim=f"{data.name or 'This commander'} was born in {data.birth_year or '[year missing]'}.",
        sources=data.birth_year_sources,
        critical=True,
        additional_instruction="Verify the birth year from the cited page. If the page provides a full birthdate but the year matches, consider it supported."
    )

    # commander_career_start
    career_group = evaluator.add_parallel(
        id="commander_career_start",
        desc="Provide the year the commander joined the Border Patrol with supporting reference",
        parent=bp_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.joined_year),
        id="factual_accuracy_career_start",
        desc="The year the commander joined Border Patrol is correctly provided (non-empty)",
        parent=career_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_career_start",
        desc="Valid reference URL is provided confirming the career start year",
        parent=career_group,
        claim=f"{data.name or 'This commander'} joined the U.S. Border Patrol in {data.joined_year or '[year missing]'}.",
        sources=data.joined_year_sources,
        critical=True,
        additional_instruction="Look for a biography or official profile confirming the join year."
    )

    # commander_appointment_date
    appoint_group = evaluator.add_parallel(
        id="commander_appointment_date",
        desc="Provide the month and year when the commander was appointed as commander-at-large with supporting reference",
        parent=bp_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.appointment_month_year),
        id="factual_accuracy_appointment",
        desc="The appointment date (month and year) is correctly provided (non-empty)",
        parent=appoint_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_appointment",
        desc="Valid reference URL is provided confirming the appointment date",
        parent=appoint_group,
        claim=f"{data.name or 'This commander'} was appointed as commander-at-large in {data.appointment_month_year or '[month/year missing]'}.",
        sources=data.appointment_sources,
        critical=True,
        additional_instruction="Confirm both the role (commander-at-large) and the month/year of appointment."
    )

    # commander_prior_position
    prior_group = evaluator.add_parallel(
        id="commander_prior_position",
        desc="Identify the specific Border Patrol sector the commander led before becoming commander-at-large with supporting reference",
        parent=bp_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.prior_sector),
        id="factual_accuracy_prior_position",
        desc="The specific Border Patrol sector is correctly identified (non-empty)",
        parent=prior_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_prior_position",
        desc="Valid reference URL is provided confirming the prior position",
        parent=prior_group,
        claim=f"Before being appointed commander-at-large, {data.name or 'this commander'} led the {data.prior_sector or '[sector missing]'} Sector of the U.S. Border Patrol.",
        sources=data.prior_sector_sources,
        critical=True,
        additional_instruction="Accept formulations like 'Chief of the [Sector] Sector'. The source must clearly indicate sector leadership."
    )


async def build_fifth_circuit_tree(
    evaluator: Evaluator,
    parent,
    data: FifthCircuitExtraction
) -> None:
    fc_node = evaluator.add_parallel(
        id="fifth_circuit_decision",
        desc="Identify the Fifth Circuit Court of Appeals decision on immigration detention policy in early 2026",
        parent=parent,
        critical=False
    )

    # decision_date
    date_group = evaluator.add_parallel(
        id="decision_date",
        desc="Provide the exact date (month, day, year) of the Fifth Circuit panel decision with supporting reference",
        parent=fc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.decision_date),
        id="factual_accuracy_decision_date",
        desc="The exact date of the decision is correctly provided (non-empty)",
        parent=date_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_decision_date",
        desc="Valid reference URL is provided confirming the decision date",
        parent=date_group,
        claim=f"The Fifth Circuit panel issued its decision on the Trump administration's mandatory immigration detention policy on {data.decision_date or '[date missing]'}.",
        sources=data.decision_date_sources,
        critical=True,
        additional_instruction="Ensure the page is about the relevant Fifth Circuit panel decision and confirms the exact date."
    )

    # majority_author
    maj_group = evaluator.add_parallel(
        id="majority_author",
        desc="Identify the judge who wrote the majority opinion with supporting reference",
        parent=fc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.majority_author),
        id="factual_accuracy_majority_author",
        desc="The majority opinion author is correctly identified (non-empty)",
        parent=maj_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_majority_author",
        desc="Valid reference URL is provided confirming the majority opinion author",
        parent=maj_group,
        claim=f"The majority opinion in this Fifth Circuit decision was written by Judge {data.majority_author or '[name missing]'}.",
        sources=data.majority_author_sources,
        critical=True,
        additional_instruction="Confirm authorship of the majority opinion on the cited page."
    )

    # appointing_president_majority
    appt_group = evaluator.add_parallel(
        id="appointing_president_majority",
        desc="Identify which U.S. President appointed the majority opinion author with supporting reference",
        parent=fc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.appointing_president),
        id="factual_accuracy_appointing_president",
        desc="The appointing president is correctly identified (non-empty)",
        parent=appt_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_appointing_president",
        desc="Valid reference URL is provided confirming the appointing president",
        parent=appt_group,
        claim=f"Judge {data.majority_author or '[majority author]'} was appointed (nominated/commissioned) by President {data.appointing_president or '[president missing]'}.",
        sources=data.appointing_president_sources,
        critical=True,
        additional_instruction="Treat 'nominated by' or 'appointed by' as equivalent for federal judicial appointment."
    )

    # dissenting_judge
    diss_group = evaluator.add_parallel(
        id="dissenting_judge",
        desc="Identify the judge who dissented in the decision with supporting reference",
        parent=fc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.dissenting_judge),
        id="factual_accuracy_dissenting_judge",
        desc="The dissenting judge is correctly identified (non-empty)",
        parent=diss_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_dissenting_judge",
        desc="Valid reference URL is provided confirming the dissenting judge",
        parent=diss_group,
        claim=f"Judge {data.dissenting_judge or '[name missing]'} dissented in this Fifth Circuit decision.",
        sources=data.dissenting_judge_sources,
        critical=True,
        additional_instruction="The page should clearly indicate that this judge wrote or joined a dissent."
    )

    # prior_judicial_opposition_count
    count_group = evaluator.add_parallel(
        id="prior_judicial_opposition_count",
        desc="Provide the approximate number of federal judges who had ruled against the administration's detention policy before this Fifth Circuit decision with supporting reference",
        parent=fc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.prior_opposition_count),
        id="factual_accuracy_opposition_count",
        desc="The approximate count of opposing judges is correctly provided (non-empty)",
        parent=count_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_opposition_count",
        desc="Valid reference URL is provided confirming the count of opposing judges",
        parent=count_group,
        claim=f"Before this Fifth Circuit decision, approximately {data.prior_opposition_count or '[count missing]'} federal judges had ruled against the administration's detention policy.",
        sources=data.prior_opposition_count_sources,
        critical=True,
        additional_instruction="Treat approximate phrasings like 'about', 'roughly', or ranges as acceptable if consistent with the stated number."
    )


async def build_scotus_2025_tree(
    evaluator: Evaluator,
    parent,
    data: SupremeCourt2025Extraction
) -> None:
    sc_node = evaluator.add_parallel(
        id="supreme_court_2025_decision",
        desc="Identify the April 2025 Supreme Court decision regarding immigration detention and removal",
        parent=parent,
        critical=False
    )

    # decision_date_scotus_2025
    sc_date_group = evaluator.add_parallel(
        id="decision_date_scotus_2025",
        desc="Provide the exact date (month, day, year) of the Supreme Court decision with supporting reference",
        parent=sc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.decision_date),
        id="factual_accuracy_scotus_date",
        desc="The exact date of the Supreme Court decision is correctly provided (non-empty)",
        parent=sc_date_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_scotus_2025_date",
        desc="Valid reference URL is provided confirming the decision date",
        parent=sc_date_group,
        claim=f"The U.S. Supreme Court issued its immigration detention/removal decision on {data.decision_date or '[date missing]'} in April 2025.",
        sources=data.decision_date_sources,
        critical=True,
        additional_instruction="Ensure the page concerns the April 2025 Supreme Court decision addressing immigration detention/removal and confirms the exact date."
    )

    # case_name_2025
    case_group = evaluator.add_parallel(
        id="case_name_2025",
        desc="Provide the case name for the April 2025 Supreme Court decision with supporting reference",
        parent=sc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.case_name),
        id="factual_accuracy_case_name",
        desc="The case name is correctly provided (non-empty)",
        parent=case_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_case_name_2025",
        desc="Valid reference URL is provided confirming the case name",
        parent=case_group,
        claim=f"The case name for the April 2025 Supreme Court immigration decision is {data.case_name or '[case name missing]'}.",
        sources=data.case_name_sources,
        critical=True,
        additional_instruction="The page must clearly show the official case caption/name."
    )

    # vote_split_2025
    vote_group = evaluator.add_parallel(
        id="vote_split_2025",
        desc="Provide the vote split for the April 2025 decision with supporting reference",
        parent=sc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.vote_split),
        id="factual_accuracy_vote_split",
        desc="The vote split is correctly provided (non-empty)",
        parent=vote_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_vote_split_2025",
        desc="Valid reference URL is provided confirming the vote split",
        parent=vote_group,
        claim=f"The vote split for the April 2025 decision was {data.vote_split or '[vote split missing]'}.",
        sources=data.vote_split_sources,
        critical=True,
        additional_instruction="Accept standard formats like '6-3' or '5–4'."
    )

    # barrett_position_2025
    barrett_group = evaluator.add_parallel(
        id="barrett_position_2025",
        desc="Identify whether Justice Amy Coney Barrett was in the majority or dissent with supporting reference",
        parent=sc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.barrett_position),
        id="factual_accuracy_barrett_position",
        desc="Justice Barrett's position (majority or dissent) is correctly identified (non-empty)",
        parent=barrett_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_barrett_2025",
        desc="Valid reference URL is provided confirming Justice Barrett's position",
        parent=barrett_group,
        claim=f"Justice Amy Coney Barrett was in the {data.barrett_position or '[position missing]'} in this April 2025 decision.",
        sources=data.barrett_position_sources,
        critical=True,
        additional_instruction="The cited page should explicitly indicate if Barrett joined the majority or dissent."
    )


async def build_venezuela_tree(
    evaluator: Evaluator,
    parent,
    data: VenezuelaTimelineExtraction
) -> None:
    vz_node = evaluator.add_parallel(
        id="venezuela_operation",
        desc="Identify key dates and facts about U.S. military operations and diplomatic actions regarding Venezuela in 2025-2026",
        parent=parent,
        critical=False
    )

    # caracas_strike_date
    strike_group = evaluator.add_parallel(
        id="caracas_strike_date",
        desc="Provide the exact date (month, day, year) when U.S. forces carried out strikes on Caracas with supporting reference",
        parent=vz_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.strike_date),
        id="factual_accuracy_strike_date",
        desc="The exact date of the Caracas strikes is correctly provided (non-empty)",
        parent=strike_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_strike_date",
        desc="Valid reference URL is provided confirming the strike date",
        parent=strike_group,
        claim=f"U.S. forces carried out military strikes on Caracas, Venezuela on {data.strike_date or '[date missing]'}.",
        sources=data.strike_date_sources,
        critical=True,
        additional_instruction="If the event did not occur or is not supported by reputable sources, return Not Supported."
    )

    # maduro_capture
    capture_group = evaluator.add_parallel(
        id="maduro_capture",
        desc="Confirm that Nicolás Maduro was captured during the operation with supporting reference",
        parent=vz_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.maduro_captured),
        id="factual_accuracy_maduro_capture",
        desc="Confirmation that Maduro was captured is correctly provided (non-empty yes/no statement)",
        parent=capture_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_maduro_capture",
        desc="Valid reference URL is provided confirming Maduro's capture",
        parent=capture_group,
        claim="Nicolás Maduro was captured during the described operation.",
        sources=data.maduro_captured_sources,
        critical=True,
        additional_instruction="Only mark as supported if the page clearly states Maduro was captured."
    )

    # interim_leader
    interim_group = evaluator.add_parallel(
        id="interim_leader",
        desc="Identify the person who became Venezuela's acting/interim president after Maduro's capture with supporting reference",
        parent=vz_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.interim_leader),
        id="factual_accuracy_interim_leader",
        desc="The interim leader is correctly identified (non-empty)",
        parent=interim_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_interim_leader",
        desc="Valid reference URL is provided confirming the interim leader",
        parent=interim_group,
        claim=f"Following Maduro's capture, {data.interim_leader or '[name missing]'} became Venezuela's acting or interim president.",
        sources=data.interim_leader_sources,
        critical=True,
        additional_instruction="The page must explicitly identify the interim or acting president."
    )

    # diplomatic_restoration_date
    diplo_group = evaluator.add_parallel(
        id="diplomatic_restoration_date",
        desc="Provide the exact date (month, day, year) when the U.S. and Venezuela agreed to reestablish diplomatic relations with supporting reference",
        parent=vz_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.diplomatic_restoration_date),
        id="factual_accuracy_diplomatic_date",
        desc="The exact date of diplomatic restoration is correctly provided (non-empty)",
        parent=diplo_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_diplomatic_date",
        desc="Valid reference URL is provided confirming the diplomatic restoration date",
        parent=diplo_group,
        claim=f"The U.S. and Venezuela agreed to reestablish diplomatic relations on {data.diplomatic_restoration_date or '[date missing]'}.",
        sources=data.diplomatic_restoration_date_sources,
        critical=True,
        additional_instruction="The page should clearly state the agreement to reestablish diplomatic relations and the exact date."
    )

    # blockade_order_date
    blockade_group = evaluator.add_parallel(
        id="blockade_order_date",
        desc="Provide the exact date (month, day, year) when President Trump ordered a naval blockade of Venezuelan oil tankers with supporting reference",
        parent=vz_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(data.blockade_order_date),
        id="factual_accuracy_blockade_date",
        desc="The exact date of the blockade order is correctly provided (non-empty)",
        parent=blockade_group,
        critical=True
    )
    await _add_and_verify_with_sources(
        evaluator,
        node_id="source_verification_blockade_date",
        desc="Valid reference URL is provided confirming the blockade order date",
        parent=blockade_group,
        claim=f"President Trump ordered a naval blockade of Venezuelan oil tankers on {data.blockade_order_date or '[date missing]'}.",
        sources=data.blockade_order_date_sources,
        critical=True,
        additional_instruction="Only mark as supported if the page explicitly states an order for a naval blockade and the date."
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
    Evaluate an answer for the multi-part immigration and Venezuela relations task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator is parallel across parts
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

    # Extract information for each part
    border_data, fifth_data, scotus_data, vz_data = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_border_patrol_commander(),
            template_class=BorderPatrolCommanderExtraction,
            extraction_name="border_patrol_commander_extraction"
        ),
        evaluator.extract(
            prompt=prompt_extract_fifth_circuit(),
            template_class=FifthCircuitExtraction,
            extraction_name="fifth_circuit_extraction"
        ),
        evaluator.extract(
            prompt=prompt_extract_scotus_2025(),
            template_class=SupremeCourt2025Extraction,
            extraction_name="scotus_apr_2025_extraction"
        ),
        evaluator.extract(
            prompt=prompt_extract_venezuela(),
            template_class=VenezuelaTimelineExtraction,
            extraction_name="venezuela_timeline_extraction"
        )
    )

    # Build verification subtrees
    await asyncio.gather(
        build_border_patrol_commander_tree(evaluator, root, border_data),
        build_fifth_circuit_tree(evaluator, root, fifth_data),
        build_scotus_2025_tree(evaluator, root, scotus_data),
        build_venezuela_tree(evaluator, root, vz_data)
    )

    return evaluator.get_summary()