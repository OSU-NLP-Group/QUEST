import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_elections_2026_window"
TASK_DESCRIPTION = """
Identify at least two, and up to four, significant political elections that occurred in the United States between January 20, 2026 and March 10, 2026. Each election must be either a special election runoff or a congressional primary election. The elections you identify must collectively meet all of these criteria:

1. At least one election occurred in Texas
2. At least one election was a special election runoff (not just a special election, but specifically the runoff round)
3. At least one election resulted in a victory by a candidate whose party was the minority party in that district according to the 2024 presidential election results

For each election, provide all of the following information:
- The exact date the election was held
- The state and the specific legislative or congressional district being contested
- The type of election (special election runoff or congressional primary)
- The complete names and party affiliations of all major candidates who received votes in this election
- The vote totals or vote percentages received by each major candidate
- The name and party affiliation of the winning candidate (or note if the election advanced to a runoff with no outright winner)
- Whether this election resulted in a change in party control of the seat
- At least one reference URL from a credible news source documenting the election results

Additionally, for each election that was a special election runoff:
- Provide the reason why the seat became vacant (resignation, appointment to other position, death, etc.)
- Provide the name and party of the person who previously held the seat
- Provide the date when the previous seat holder vacated the position

Additionally, for each election where the winner's party was the minority party in that district based on 2024 presidential results:
- State the margin (in percentage points) by which the opposing party's presidential candidate carried that district in the November 2024 presidential election
- Provide a reference showing this presidential election result for the district
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CandidateInfo(BaseModel):
    name: Optional[str] = None
    party: Optional[str] = None
    result: Optional[str] = None  # votes or percentage, keep as free-form string


class RunoffDetails(BaseModel):
    vacancy_reason: Optional[str] = None  # e.g., resignation, death, appointment, etc.
    previous_holder_name: Optional[str] = None
    previous_holder_party: Optional[str] = None
    vacancy_date: Optional[str] = None  # keep as string, we'll not parse strict formats here


class MinorityDetails(BaseModel):
    is_minority_win: Optional[bool] = None
    opposing_party_margin_2024: Optional[str] = None  # keep as string, e.g., "R +6.5" or "D +4"
    presidential_result_sources: List[str] = Field(default_factory=list)


class ElectionRecord(BaseModel):
    date: Optional[str] = None  # keep string for flexible formats; we'll parse range separately
    state: Optional[str] = None
    district: Optional[str] = None  # e.g., "TX-02", "Texas 2nd Congressional District"
    election_type: Optional[str] = None  # "special election runoff" or "congressional primary"
    candidates: List[CandidateInfo] = Field(default_factory=list)
    winner_name: Optional[str] = None
    winner_party: Optional[str] = None
    advanced_to_runoff: Optional[bool] = None  # True if no outright winner and advanced to runoff
    party_flip: Optional[str] = None  # "yes"/"no"/"unknown" (presence is required)
    sources: List[str] = Field(default_factory=list)  # at least one credible news URL is required
    runoff_details: Optional[RunoffDetails] = None
    minority_details: Optional[MinorityDetails] = None


class ElectionsExtraction(BaseModel):
    elections: List[ElectionRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_elections() -> str:
    return """
    Extract from the answer a structured list of elections (between 2 and 6 items if present in the answer; preserve the original order). For each election mentioned, return an object with the following fields:

    Required core fields for every election:
    - date: the exact calendar date the election was held, as stated in the answer (string)
    - state: the U.S. state name or postal abbreviation (string)
    - district: the specific legislative or congressional district or seat (string, e.g., "TX-02", "Texas 2nd Congressional District", "California 20th Congressional District")
    - election_type: one of ["special election runoff", "congressional primary"] exactly as characterized in the answer (string)
    - candidates: array of candidate objects, each with:
        - name (string)
        - party (string, e.g., "Republican", "Democratic", "Libertarian", etc.)
        - result (string for votes or percentages, e.g., "12,345 (54.1%)" or "54.1%")
    - winner_name: the winner's full name if known; otherwise null if there was no outright winner
    - winner_party: the winner's party if known; otherwise null if no winner
    - advanced_to_runoff: boolean value: true if the election advanced to a runoff with no outright winner; false otherwise; null if not stated
    - party_flip: whether there was a change in party control of the seat as stated in the answer (string such as "yes", "no", "not applicable", or "unknown")
    - sources: array of one or more credible news URLs cited for this election's results. Extract exactly the URLs explicitly provided in the answer. If none are given, set to an empty array.

    Additional fields if (and only if) the election is a special election runoff:
    - runoff_details: an object with:
        - vacancy_reason: why the seat became vacant (e.g., resignation, death, appointment). Null if not stated.
        - previous_holder_name: the name of the person who previously held the seat. Null if not stated.
        - previous_holder_party: that person's party. Null if not stated.
        - vacancy_date: the date the previous holder vacated the seat. Null if not stated.

    Additional fields if (and only if) the answer claims the winner's party was the minority party in the district based on 2024 presidential results:
    - minority_details: an object with:
        - is_minority_win: boolean (true if the answer claims the winner's party was the district's minority party based on 2024 results; false otherwise; null if not stated)
        - opposing_party_margin_2024: the stated margin by which the opposing party's presidential candidate carried the district in 2024 (string, allow formats like "R +6.5", "D +4", "Republican +6.5", "Democratic +4"). Null if not provided.
        - presidential_result_sources: array of URLs that show the 2024 presidential results for this district. Empty array if none provided.

    General rules:
    - Do not invent or infer. Only extract info explicitly present in the answer.
    - If any requested field is missing, set it to null (or empty list where appropriate).
    - Return JSON in the schema exactly as specified.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
DATE_RANGE_START = datetime(2026, 1, 20)
DATE_RANGE_END = datetime(2026, 3, 10)
ALLOWED_TYPES = {"special election runoff", "congressional primary"}


def _parse_date_maybe(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    date_str = date_str.strip()
    fmts = [
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y/%m/%d",
        "%d %B %Y",
        "%d %b %Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except Exception:
            continue
    # Last resort: try to let Python parse month-day-year patterns in a forgiving way
    try:
        # naive fallback for forms like "March 5 2026"
        parts = date_str.replace(",", "").split()
        if len(parts) == 3:
            try:
                return datetime.strptime(" ".join(parts), "%B %d %Y")
            except Exception:
                pass
    except Exception:
        pass
    return None


def _state_mentions_texas(state: Optional[str], district: Optional[str]) -> bool:
    s = f"{state or ''} {district or ''}".lower()
    return ("texas" in s) or (" tx" in s) or (s.startswith("tx-")) or ("tx-" in s)


def _is_runoff_type(e: ElectionRecord) -> bool:
    t = (e.election_type or "").strip().lower()
    return "runoff" in t and "special" in t  # explicitly special election runoff


def _is_allowed_type(e: ElectionRecord) -> bool:
    t = (e.election_type or "").strip().lower()
    return t in ALLOWED_TYPES


def _minority_claimed(e: ElectionRecord) -> bool:
    if e.minority_details and e.minority_details.is_minority_win is not None:
        return bool(e.minority_details.is_minority_win)
    return False


def _first_n_elections(elections: List[ElectionRecord], n: int = 4) -> List[ElectionRecord]:
    return elections[:n]


def _ordinal(n: int) -> str:
    mapping = {1: "first", 2: "second", 3: "third", 4: "fourth"}
    return mapping.get(n, f"#{n}")


def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Verification for one election                                               #
# --------------------------------------------------------------------------- #
async def verify_one_election(
    evaluator: Evaluator,
    parent_node,
    e: ElectionRecord,
    index: int,
    node_prefix: str,
) -> None:
    """
    Build verification subtree for a single election record.
    node_prefix examples: "E1", "E2", "E3", "E4"
    """
    # Election parent (parallel, non-critical to allow partial scoring per election)
    election_node = evaluator.add_parallel(
        id=f"election_{index+1}",
        desc=f"Evaluation of the { _ordinal(index+1) } election",
        parent=parent_node,
        critical=False,
    )

    # Convenience: sources for this election
    sources_list = e.sources if e.sources else []

    # E*_Date
    leaf_date = evaluator.add_leaf(
        id=f"{node_prefix}_Date",
        desc="The exact election date is provided",
        parent=election_node,
        critical=True,
    )
    date_claim = f"The election was held on {e.date}."
    await evaluator.verify(
        claim=date_claim,
        node=leaf_date,
        sources=sources_list,
        additional_instruction="Verify the stated election date exactly or within reasonable timezone publication lag.",
    )

    # E*_State
    leaf_state = evaluator.add_leaf(
        id=f"{node_prefix}_State",
        desc="The state is clearly identified",
        parent=election_node,
        critical=True,
    )
    state_claim = f"This election took place in the state of {e.state}."
    await evaluator.verify(
        claim=state_claim,
        node=leaf_state,
        sources=sources_list,
        additional_instruction="Confirm the state for the election jurisdiction as written on the page.",
    )

    # E*_District
    leaf_district = evaluator.add_leaf(
        id=f"{node_prefix}_District",
        desc="The specific district or office is clearly identified",
        parent=election_node,
        critical=True,
    )
    district_claim = f"This election was for {e.district}."
    await evaluator.verify(
        claim=district_claim,
        node=leaf_district,
        sources=sources_list,
        additional_instruction="Allow reasonable variants like 'U.S. House District 2' vs 'TX-02' that clearly refer to the same seat.",
    )

    # E*_Type
    leaf_type = evaluator.add_leaf(
        id=f"{node_prefix}_Type",
        desc="The election type (special election runoff or congressional primary) is specified",
        parent=election_node,
        critical=True,
    )
    type_claim = f"This election was a {e.election_type}."
    await evaluator.verify(
        claim=type_claim,
        node=leaf_type,
        sources=sources_list,
        additional_instruction="Specifically verify whether it is a 'special election runoff' (runoff round of a special election) or a 'congressional primary'.",
    )

    # E*_Candidate_Names
    leaf_cand_names = evaluator.add_leaf(
        id=f"{node_prefix}_Candidate_Names",
        desc="Complete names of all major candidates are provided",
        parent=election_node,
        critical=True,
    )
    cand_names_list = [c.name for c in e.candidates if _non_empty(c.name)]
    cand_names_claim = (
        f"The major candidates who received votes in this election included: {', '.join(cand_names_list)}."
        if cand_names_list else "No candidate names were provided."
    )
    await evaluator.verify(
        claim=cand_names_claim,
        node=leaf_cand_names,
        sources=sources_list,
        additional_instruction="Cross-check that the page lists these as the main vote-receiving candidates; allow minor name variants.",
    )

    # E*_Candidate_Parties
    leaf_cand_parties = evaluator.add_leaf(
        id=f"{node_prefix}_Candidate_Parties",
        desc="Party affiliations for all major candidates are provided",
        parent=election_node,
        critical=True,
    )
    party_pairs = [f"{c.name} ({c.party})" for c in e.candidates if _non_empty(c.name) and _non_empty(c.party)]
    cand_party_claim = (
        f"The party affiliations for the major candidates were: {', '.join(party_pairs)}."
        if party_pairs else "No candidate parties were provided."
    )
    await evaluator.verify(
        claim=cand_party_claim,
        node=leaf_cand_parties,
        sources=sources_list,
        additional_instruction="Confirm that each candidate is labeled with the given party affiliation.",
    )

    # E*_Vote_Results
    leaf_votes = evaluator.add_leaf(
        id=f"{node_prefix}_Vote_Results",
        desc="Vote totals or percentages for all major candidates are provided",
        parent=election_node,
        critical=True,
    )
    results_pairs = [f"{c.name}: {c.result}" for c in e.candidates if _non_empty(c.name) and _non_empty(c.result)]
    vote_claim = (
        f"The vote results were reported as follows: { '; '.join(results_pairs) }."
        if results_pairs else "No vote totals or percentages were provided for the candidates."
    )
    await evaluator.verify(
        claim=vote_claim,
        node=leaf_votes,
        sources=sources_list,
        additional_instruction="Allow reasonable rounding differences. Verify that each listed candidate has a corresponding vote total or percentage.",
    )

    # E*_Winner_Name
    leaf_winner_name = evaluator.add_leaf(
        id=f"{node_prefix}_Winner_Name",
        desc="The winning candidate's name is stated or runoff status is noted",
        parent=election_node,
        critical=True,
    )
    if e.advanced_to_runoff is True and not _non_empty(e.winner_name):
        winner_name_claim = "No candidate won outright; the election advanced to a runoff."
    elif _non_empty(e.winner_name):
        winner_name_claim = f"The winning candidate was {e.winner_name}."
    else:
        winner_name_claim = "The answer states a winner or runoff status for this election."
    await evaluator.verify(
        claim=winner_name_claim,
        node=leaf_winner_name,
        sources=sources_list,
        additional_instruction="If the page shows that no candidate cleared the threshold and a runoff was triggered, that should count as 'no outright winner'.",
    )

    # E*_Winner_Party
    leaf_winner_party = evaluator.add_leaf(
        id=f"{node_prefix}_Winner_Party",
        desc="The winning candidate's party affiliation is stated or runoff status is noted",
        parent=election_node,
        critical=True,
    )
    if e.advanced_to_runoff is True and not _non_empty(e.winner_party):
        winner_party_claim = "Because there was no outright winner, there is no single winner's party to report for this election."
    elif _non_empty(e.winner_party) and _non_empty(e.winner_name):
        winner_party_claim = f"The winner's party was {e.winner_party}."
    else:
        winner_party_claim = "The answer states the winner's party or correctly notes there was no winner due to runoff."
    await evaluator.verify(
        claim=winner_party_claim,
        node=leaf_winner_party,
        sources=sources_list,
        additional_instruction="If a winner is declared, verify the winner's party; otherwise confirm that a runoff status applies.",
    )

    # E*_Party_Flip (presence check only as per rubric wording)
    party_flip_present = _non_empty(e.party_flip)
    evaluator.add_custom_node(
        result=party_flip_present,
        id=f"{node_prefix}_Party_Flip",
        desc="Whether the election resulted in party control change is stated",
        parent=election_node,
        critical=True,
    )

    # E*_Reference (presence check for at least one credible news URL)
    has_ref = bool(sources_list)
    evaluator.add_custom_node(
        result=has_ref,
        id=f"{node_prefix}_Reference",
        desc="At least one credible news source URL is provided",
        parent=election_node,
        critical=True,
    )

    # ------------------- Conditional: Runoff details --------------------- #
    runoff_seq = evaluator.add_sequential(
        id=f"{node_prefix}_Runoff_Conditionals",
        desc="Additional details if this is a special election runoff",
        parent=election_node,
        critical=False,
    )

    # E*_Is_Runoff_Check (critical within this sequence)
    runoff_check = evaluator.add_leaf(
        id=f"{node_prefix}_Is_Runoff_Check",
        desc="Verify if this election is a special election runoff",
        parent=runoff_seq,
        critical=True,
    )
    runoff_check_claim = "This election was a special election runoff (the runoff round of a special election)."
    await evaluator.verify(
        claim=runoff_check_claim,
        node=runoff_check,
        sources=sources_list,
        additional_instruction="Confirm that the page explicitly indicates this is the runoff round of a special election (not just any special election).",
    )

    # E*_Runoff_Details (critical parallel group if above passes)
    runoff_parallel = evaluator.add_parallel(
        id=f"{node_prefix}_Runoff_Details",
        desc="Required runoff-specific details",
        parent=runoff_seq,
        critical=True,
    )

    # E*_Vacancy_Reason
    leaf_vacancy_reason = evaluator.add_leaf(
        id=f"{node_prefix}_Vacancy_Reason",
        desc="Reason for seat vacancy is provided",
        parent=runoff_parallel,
        critical=True,
    )
    vac_reason = e.runoff_details.vacancy_reason if e.runoff_details else None
    vac_reason_claim = f"The seat became vacant due to: {vac_reason}."
    await evaluator.verify(
        claim=vac_reason_claim,
        node=leaf_vacancy_reason,
        sources=sources_list,
        additional_instruction="Verify that the page states why the seat became vacant (e.g., resignation, death, appointment).",
    )

    # E*_Previous_Holder_Name
    leaf_prev_name = evaluator.add_leaf(
        id=f"{node_prefix}_Previous_Holder_Name",
        desc="Name of previous seat holder is provided",
        parent=runoff_parallel,
        critical=True,
    )
    prev_name = e.runoff_details.previous_holder_name if e.runoff_details else None
    prev_name_claim = f"The seat was previously held by {prev_name}."
    await evaluator.verify(
        claim=prev_name_claim,
        node=leaf_prev_name,
        sources=sources_list,
        additional_instruction="Verify the previous officeholder's name as presented on the cited page.",
    )

    # E*_Previous_Holder_Party
    leaf_prev_party = evaluator.add_leaf(
        id=f"{node_prefix}_Previous_Holder_Party",
        desc="Party of previous seat holder is provided",
        parent=runoff_parallel,
        critical=True,
    )
    prev_party = e.runoff_details.previous_holder_party if e.runoff_details else None
    prev_party_claim = f"The previous officeholder's party was {prev_party}."
    await evaluator.verify(
        claim=prev_party_claim,
        node=leaf_prev_party,
        sources=sources_list,
        additional_instruction="Verify the previous officeholder's party as shown on the page.",
    )

    # E*_Vacancy_Date
    leaf_vacancy_date = evaluator.add_leaf(
        id=f"{node_prefix}_Vacancy_Date",
        desc="Date when previous holder vacated is provided",
        parent=runoff_parallel,
        critical=True,
    )
    vac_date = e.runoff_details.vacancy_date if e.runoff_details else None
    vac_date_claim = f"The previous officeholder vacated the seat on {vac_date}."
    await evaluator.verify(
        claim=vac_date_claim,
        node=leaf_vacancy_date,
        sources=sources_list,
        additional_instruction="Verify the date (or approximate date if presented) the seat was vacated.",
    )

    # ---------------- Conditional: Minority party win details ------------- #
    minority_seq = evaluator.add_sequential(
        id=f"{node_prefix}_Minority_Conditionals",
        desc="Additional details if winner's party was minority in district",
        parent=election_node,
        critical=False,
    )

    # E*_Is_Minority_Check (critical within sequence)
    leaf_is_minority = evaluator.add_leaf(
        id=f"{node_prefix}_Is_Minority_Check",
        desc="Verify if winner's party was minority based on 2024 presidential results",
        parent=minority_seq,
        critical=True,
    )
    # Prefer presidential result sources for this check
    minority_sources = (e.minority_details.presidential_result_sources if (e.minority_details and e.minority_details.presidential_result_sources) else []) or sources_list
    is_minority_text = "was" if _minority_claimed(e) else "was not"
    minority_check_claim = (
        f"Based on the November 2024 presidential results for {e.district} in {e.state}, "
        f"the winner's party {is_minority_text} the minority party in that district."
    )
    await evaluator.verify(
        claim=minority_check_claim,
        node=leaf_is_minority,
        sources=minority_sources,
        additional_instruction="Focus on whether the district was carried in 2024 by the party opposing the winner's party in this 2026 election. If the provided source does not give district-level 2024 presidential results, treat the claim as unsupported.",
    )

    # E*_Minority_Details (critical parallel group if above passes)
    minority_parallel = evaluator.add_parallel(
        id=f"{node_prefix}_Minority_Details",
        desc="Required minority-win-specific details",
        parent=minority_seq,
        critical=True,
    )

    # E*_Presidential_Margin
    leaf_pres_margin = evaluator.add_leaf(
        id=f"{node_prefix}_Presidential_Margin",
        desc="Margin by which opposing presidential candidate carried district in 2024 is provided",
        parent=minority_parallel,
        critical=True,
    )
    opp_margin = e.minority_details.opposing_party_margin_2024 if e.minority_details else None
    pres_margin_claim = f"The opposing party's presidential candidate carried the district in 2024 by approximately {opp_margin}."
    await evaluator.verify(
        claim=pres_margin_claim,
        node=leaf_pres_margin,
        sources=minority_sources,
        additional_instruction="Allow rounding differences or formatting variations (e.g., 'R +6.5', 'Republican +6.5', or 'Democratic +4'). Confirm the direction and magnitude approximately.",
    )

    # E*_Presidential_Reference (presence check)
    has_pres_ref = bool(e.minority_details and e.minority_details.presidential_result_sources)
    evaluator.add_custom_node(
        result=has_pres_ref,
        id=f"{node_prefix}_Presidential_Reference",
        desc="Reference showing 2024 presidential result for district is provided",
        parent=minority_parallel,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Collection-level verifications                                              #
# --------------------------------------------------------------------------- #
def add_collection_requirement_nodes(
    evaluator: Evaluator,
    parent_node,
    elections_eval: List[ElectionRecord],
) -> None:
    req_node = evaluator.add_parallel(
        id="collection_requirements",
        desc="Verify the collection of elections meets all specified criteria",
        parent=parent_node,
        critical=True,
    )

    # Minimum_Count
    evaluator.add_custom_node(
        result=len(elections_eval) >= 2,
        id="Minimum_Count",
        desc="At least two elections are provided",
        parent=req_node,
        critical=True,
    )

    # Maximum_Count
    evaluator.add_custom_node(
        result=len(elections_eval) <= 4,
        id="Maximum_Count",
        desc="No more than four elections are provided",
        parent=req_node,
        critical=True,
    )

    # Texas_Requirement
    has_texas = any(_state_mentions_texas(e.state, e.district) for e in elections_eval)
    evaluator.add_custom_node(
        result=has_texas,
        id="Texas_Requirement",
        desc="At least one election occurred in Texas",
        parent=req_node,
        critical=True,
    )

    # Runoff_Requirement (at least one special election runoff)
    has_runoff = any(_is_runoff_type(e) for e in elections_eval)
    evaluator.add_custom_node(
        result=has_runoff,
        id="Runoff_Requirement",
        desc="At least one election was a special election runoff",
        parent=req_node,
        critical=True,
    )

    # Minority_Win_Requirement
    has_minority_win = any(_minority_claimed(e) for e in elections_eval)
    evaluator.add_custom_node(
        result=has_minority_win,
        id="Minority_Win_Requirement",
        desc="At least one election resulted in a minority party victory",
        parent=req_node,
        critical=True,
    )

    # Date_Range - all dates within Jan 20, 2026 to Mar 10, 2026
    all_in_range = True
    for e in elections_eval:
        d = _parse_date_maybe(e.date)
        if not d or d < DATE_RANGE_START or d > DATE_RANGE_END:
            all_in_range = False
            break
    evaluator.add_custom_node(
        result=all_in_range,
        id="Date_Range",
        desc="All elections occurred between January 20, 2026 and March 10, 2026",
        parent=req_node,
        critical=True,
    )

    # Type_Requirement - only allowed types
    types_ok = all(_is_allowed_type(e) for e in elections_eval)
    evaluator.add_custom_node(
        result=types_ok,
        id="Type_Requirement",
        desc="All elections are either special election runoffs or congressional primaries",
        parent=req_node,
        critical=True,
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
    Evaluate an answer for the US elections (Jan 20, 2026 – Mar 10, 2026) task.
    """
    # Initialize evaluator (root: parallel aggregation)
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

    # Extract elections from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_elections(),
        template_class=ElectionsExtraction,
        extraction_name="elections_extraction",
    )

    # Keep only the first up to 4 elections for evaluation (consistent with rubric cap)
    elections_all = extracted.elections or []
    elections_eval = _first_n_elections(elections_all, 4)

    # Record a brief custom info about counts
    evaluator.add_custom_info(
        {
            "original_elections_count": len(elections_all),
            "evaluated_elections_count": len(elections_eval),
            "date_window_start": DATE_RANGE_START.isoformat(),
            "date_window_end": DATE_RANGE_END.isoformat(),
        },
        info_type="meta",
        info_name="collection_meta",
    )

    # Collection-level requirement checks
    add_collection_requirement_nodes(evaluator, root, elections_eval)

    # Per-election checks
    # Election 1 and 2 are required by rubric; 3 and 4 are optional "if provided"
    for idx, e in enumerate(elections_eval[:2]):
        await verify_one_election(
            evaluator=evaluator,
            parent_node=root,
            e=e,
            index=idx,
            node_prefix=f"E{idx+1}",
        )

    # Optional: third and fourth if provided
    if len(elections_eval) >= 3:
        await verify_one_election(
            evaluator=evaluator,
            parent_node=root,
            e=elections_eval[2],
            index=2,
            node_prefix="E3",
        )
    if len(elections_eval) >= 4:
        await verify_one_election(
            evaluator=evaluator,
            parent_node=root,
            e=elections_eval[3],
            index=3,
            node_prefix="E4",
        )

    # Return the final structured summary
    return evaluator.get_summary()