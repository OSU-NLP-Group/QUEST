import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "recent_federal_officials_2024_2025"
TASK_DESCRIPTION = (
    "Identify at least three U.S. federal officials who were appointed to or elected to their current federal positions "
    "between June 1, 2024, and November 30, 2025. The set of officials you identify must satisfy the following requirements:\n\n"
    "1. Geographic diversity: The officials must represent or have previously served in at least three different U.S. states.\n\n"
    "2. Position type diversity: Your set must include:\n"
    "   - At least one official who was elected to their federal position (such as U.S. Senate or U.S. House of Representatives)\n"
    "   - At least one official who was appointed to their federal position (such as FBI leadership, Cabinet positions, federal agency leadership, or other presidential appointments)\n\n"
    "For each official, provide the following information:\n"
    "   - Their full legal name as used in official documentation\n"
    "   - The exact title of the position they held immediately before their current federal position\n"
    "   - The U.S. state where they previously served or are now representing\n"
    "   - The exact title of their current federal position\n"
    "   - The official date of their appointment or election to the federal position\n"
    "   - The date they assumed office (if different from the appointment/election date)\n"
    "   - A URL from an official government source, major news organization, or official campaign/organization website that documents their appointment or election\n"
    "   - Any historic significance of their appointment or election (e.g., first person from a particular demographic group to hold the position, if applicable)\n\n"
    "Verification: Ensure that all appointments or elections occurred within the specified time period from June 1, 2024, through November 30, 2025."
)

DATE_WINDOW_START_STR = "June 1, 2024"
DATE_WINDOW_END_STR = "November 30, 2025"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OfficialItem(BaseModel):
    full_name: Optional[str] = None
    previous_position_title: Optional[str] = None
    state: Optional[str] = None
    current_federal_position_title: Optional[str] = None
    # Election/appointment method, ideally exactly "elected" or "appointed"; else "unknown"
    selection_method: Optional[str] = None
    appointment_or_election_date: Optional[str] = None
    assumed_office_date: Optional[str] = None
    verification_urls: List[str] = Field(default_factory=list)
    historic_significance: Optional[str] = None


class CandidateOfficialsExtraction(BaseModel):
    officials: List[OfficialItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_officials() -> str:
    return """
Extract up to five candidate U.S. federal officials mentioned in the answer (limit to the first five if more are present). 
For each official, extract the following fields EXACTLY as presented in the answer:

- full_name: The official's full legal name as used in official documentation. 
- previous_position_title: The exact title of the position the person held immediately before the current federal position.
- state: The U.S. state where the official previously served or is now representing (e.g., 'Arizona', 'New York').
- current_federal_position_title: The exact title of the official's current federal position (must be a U.S. federal office/role).
- selection_method: One of "elected", "appointed", or "unknown". Use "elected" if it clearly states they were elected (e.g., U.S. Senator/Representative elections). 
  Use "appointed" if clearly a presidential/agency appointment or direct federal appointment. If not explicit and not obviously inferable from the provided titles, use "unknown".
- appointment_or_election_date: The exact date (as written in the answer) when they were appointed or elected to their current federal position.
- assumed_office_date: If the person assumed office on a different date than the appointment/election date and that date is given, provide it; otherwise return null.
- verification_urls: All URLs provided in the answer that document the appointment/election event. Only include valid URLs explicitly present in the answer (e.g., .gov pages, official campaign/organization sites, or major news organizations). Extract all such URLs; do not invent.
- historic_significance: If the answer explicitly states any historically significant aspect (e.g., 'first X to serve in Y role'), include the statement verbatim; otherwise return null.

Return a JSON object with a single key 'officials' that is an array of up to five such objects. 
If any field is missing for a given official, set it to null (or an empty array for verification_urls).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _normalize_selection_method(s: Optional[str]) -> str:
    if not s:
        return "unknown"
    s_l = s.strip().lower()
    if "elect" in s_l:
        return "elected"
    if "appoint" in s_l:
        return "appointed"
    return "unknown"


def _should_verify_assumed_office(assumed: Optional[str], appointed_or_elected_date: Optional[str]) -> bool:
    if not _is_nonempty(assumed):
        return False
    if not _is_nonempty(appointed_or_elected_date):
        return True
    return assumed.strip().lower() != appointed_or_elected_date.strip().lower()


def _first_k(items: List[Any], k: int) -> List[Any]:
    return items[:k] if items else []


def _urls_str(urls: List[str]) -> str:
    if not urls:
        return ""
    return "; ".join(urls)


# --------------------------------------------------------------------------- #
# Verification per official                                                   #
# --------------------------------------------------------------------------- #
async def verify_official(
    evaluator: Evaluator,
    parent_node,
    official: OfficialItem,
    idx: int,
) -> Tuple[bool, Optional[str], str]:
    """
    Build verification nodes for a single official and run verifications.
    Returns (qualifies, state, selection_method_normalized)
    Qualifies = all per-official critical checks pass.
    """
    idx1 = idx + 1

    official_node = evaluator.add_parallel(
        id=f"official_{idx1}",
        desc=f"Documentation bundle for candidate official #{idx1}.",
        parent=parent_node,
        critical=False  # Non-critical at item level; Set-level constraints will enforce counts/diversity
    )

    # Critical existence checks (custom nodes)
    name_exists_node = evaluator.add_custom_node(
        result=_is_nonempty(official.full_name),
        id=f"official_{idx1}_name_exists",
        desc="Full legal name is provided.",
        parent=official_node,
        critical=True
    )

    prev_pos_exists_node = evaluator.add_custom_node(
        result=_is_nonempty(official.previous_position_title),
        id=f"official_{idx1}_prev_position_exists",
        desc="Previous position title is provided.",
        parent=official_node,
        critical=True
    )

    state_exists_node = evaluator.add_custom_node(
        result=_is_nonempty(official.state),
        id=f"official_{idx1}_state_exists",
        desc="State jurisdiction is provided.",
        parent=official_node,
        critical=True
    )

    current_pos_exists_node = evaluator.add_custom_node(
        result=_is_nonempty(official.current_federal_position_title),
        id=f"official_{idx1}_current_position_exists",
        desc="Current federal position title is provided.",
        parent=official_node,
        critical=True
    )

    date_exists_node = evaluator.add_custom_node(
        result=_is_nonempty(official.appointment_or_election_date),
        id=f"official_{idx1}_date_exists",
        desc="Appointment/Election date is provided.",
        parent=official_node,
        critical=True
    )

    urls_exist_node = evaluator.add_custom_node(
        result=bool(official.verification_urls and len(official.verification_urls) > 0),
        id=f"official_{idx1}_urls_exist",
        desc="At least one verification URL is provided.",
        parent=official_node,
        critical=True
    )

    # Leaf: Source type allowed (domain/category check)
    source_allowed_node = evaluator.add_leaf(
        id=f"official_{idx1}_source_type_allowed",
        desc="Verification URL(s) are from allowed sources (official government, major news, or official campaign/organization).",
        parent=official_node,
        critical=True
    )
    source_allowed_claim = (
        "Assess whether the following URL(s) are from allowed sources: "
        f"{_urls_str(official.verification_urls)}. "
        "Allowed sources include official U.S. government domains (.gov or .mil), major news organizations "
        "(e.g., AP, Reuters, NPR, major national newspapers/networks), or the official campaign/organization website of the official. "
        "Return Correct if all provided URLs fall into allowed categories; otherwise Incorrect."
    )

    # Leaf: Appointment/Election event + date in required window, supported by sources
    method = _normalize_selection_method(official.selection_method)
    method_phrase = "appointed or elected" if method == "unknown" else ("appointed" if method == "appointed" else "elected")

    event_supported_node = evaluator.add_leaf(
        id=f"official_{idx1}_event_supported_in_range",
        desc="Appointment/election to current federal position on the stated date is supported by the provided source(s) and the date is within the specified window.",
        parent=official_node,
        critical=True
    )
    event_claim = (
        f"{official.full_name} was {method_phrase} to the position '{official.current_federal_position_title}' "
        f"on {official.appointment_or_election_date}. "
        f"This date falls between {DATE_WINDOW_START_STR} and {DATE_WINDOW_END_STR} (inclusive)."
    )
    event_additional_instruction = (
        "Verify two things from the page(s): "
        "1) The event actually happened (the person was appointed or elected to the stated current federal position), and "
        "2) The specific date provided matches and lies within the window "
        f"{DATE_WINDOW_START_STR} through {DATE_WINDOW_END_STR}. "
        "If a URL is irrelevant or does not confirm both aspects, treat it as not supporting the claim."
    )

    # Optional leaf: Assumed office date (if present and different)
    optional_nodes: List[Tuple[str, Any]] = []  # store (purpose, node) to include in 'must-pass'? No, optional.
    if _should_verify_assumed_office(official.assumed_office_date, official.appointment_or_election_date):
        assumed_node = evaluator.add_leaf(
            id=f"official_{idx1}_assumed_office_supported",
            desc="Assumed office date (if provided and different) is supported by the provided source(s).",
            parent=official_node,
            critical=False  # Non-critical: omission acceptable if not different
        )
        assumed_claim = (
            f"After being {method_phrase}, {official.full_name} assumed office on {official.assumed_office_date}."
        )
        assumed_instruction = (
            "Verify that the page explicitly states or clearly implies the provided 'assumed office' date. "
            "If the page only lists an appointment/election date but not an assumption of office, then this claim is unsupported."
        )
        optional_nodes.append(("assumed", assumed_node))
    else:
        # If not provided or same date, we consider requirement satisfied with a custom pass (non-critical)
        evaluator.add_custom_node(
            result=True,
            id=f"official_{idx1}_assumed_office_requirement_satisfied",
            desc="Assumed-office date requirement satisfied (either not different or not applicable).",
            parent=official_node,
            critical=False
        )

    # Optional leaf: Historic significance, if provided
    if _is_nonempty(official.historic_significance):
        hist_node = evaluator.add_leaf(
            id=f"official_{idx1}_historic_significance_supported",
            desc="Historic significance statement (if provided) is supported by the provided source(s).",
            parent=official_node,
            critical=False  # Non-critical since it is 'if applicable'
        )
        hist_claim = (
            f"The following historical significance statement regarding {official.full_name}'s appointment/election is supported by the page(s): "
            f"'{official.historic_significance}'."
        )
        hist_instruction = (
            "Verify that the provided page(s) explicitly support this historical significance claim (e.g., a documented 'first'). "
            "If it is not mentioned or the support is unclear, treat as unsupported."
        )
        optional_nodes.append(("historic", hist_node))
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"official_{idx1}_historic_significance_not_applicable",
            desc="Historic significance not provided; not applicable.",
            parent=official_node,
            critical=False
        )

    # Batch verifications
    claims_and_sources = []

    # Source allowed classification (simple verification, no URLs opened)
    claims_and_sources.append((
        source_allowed_claim,
        None,  # simple verify
        source_allowed_node,
        "Judge only whether each domain is an allowed source category as defined. Do not check content."
    ))

    # Event/date-in-range verification (URL verification)
    claims_and_sources.append((
        event_claim,
        official.verification_urls,
        event_supported_node,
        event_additional_instruction
    ))

    # Optional leaves
    for purpose, node in optional_nodes:
        if purpose == "assumed":
            claims_and_sources.append((
                assumed_claim,
                official.verification_urls,
                node,
                assumed_instruction
            ))
        elif purpose == "historic":
            claims_and_sources.append((
                hist_claim,
                official.verification_urls,
                node,
                hist_instruction
            ))

    # Execute verifications in parallel for this official
    if claims_and_sources:
        await evaluator.batch_verify(claims_and_sources)

    # Determine if this official "qualifies" (all critical per-official checks passed)
    must_pass_nodes = [
        name_exists_node,
        prev_pos_exists_node,
        state_exists_node,
        current_pos_exists_node,
        date_exists_node,
        urls_exist_node,
        source_allowed_node,
        event_supported_node,
    ]
    qualifies = all(n.status == "passed" for n in must_pass_nodes)

    # Return tuple for set-level constraints
    return qualifies, official.state, _normalize_selection_method(official.selection_method)


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
    Evaluate an answer for the recent federal officials identification task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel so set-level checks are not skipped by partial item failures
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

    # Top-level nodes
    candidate_node = evaluator.add_parallel(
        id="Candidate_Officials_Documented",
        desc="Provide documentation for up to five candidate officials (to allow 'at least 3' without evaluating >5 items).",
        parent=root,
        critical=False  # Non-critical: we allow partial success here
    )
    constraints_node = evaluator.add_parallel(
        id="Set_Level_Constraints",
        desc="Constraints that must be satisfied by the set of qualifying officials.",
        parent=root,
        critical=True  # Critical: final pass depends on set-level constraints
    )

    # 1) Extraction
    extracted: CandidateOfficialsExtraction = await evaluator.extract(
        prompt=prompt_extract_officials(),
        template_class=CandidateOfficialsExtraction,
        extraction_name="candidate_officials"
    )

    officials = _first_k(extracted.officials, 5)
    while len(officials) < 3:
        # Pad with empty placeholders so we always have at least 3 positions to evaluate
        officials.append(OfficialItem())

    # 2) Per-official verifications
    qualify_flags: List[bool] = []
    qualify_states: List[str] = []
    qualify_types: List[str] = []  # "elected"/"appointed"/"unknown"

    for i, off in enumerate(officials):
        qualifies, state, sel_type = await verify_official(evaluator, candidate_node, off, i)
        qualify_flags.append(qualifies)
        qualify_states.append(state or "")
        qualify_types.append(sel_type)

    # 3) Set-level constraints
    # 3.1 At least three qualifying officials
    qualifying_indices = [i for i, q in enumerate(qualify_flags) if q]
    at_least_three = len(qualifying_indices) >= 3
    evaluator.add_custom_node(
        result=at_least_three,
        id="At_Least_Three_Qualifying_Officials",
        desc="Verify that at least three candidate officials satisfy all per-official critical requirements.",
        parent=constraints_node,
        critical=True
    )

    # 3.2 Geographic diversity: At least three different states among qualifying officials
    qualifying_states_set = set(
        s.strip().lower() for i, s in enumerate(qualify_states) if i in qualifying_indices and _is_nonempty(s)
    )
    geo_diverse = len(qualifying_states_set) >= 3
    evaluator.add_custom_node(
        result=geo_diverse,
        id="Geographic_Diversity_At_Least_Three_States",
        desc="Verify that the qualifying officials collectively cover at least three different U.S. states (based on the provided state field).",
        parent=constraints_node,
        critical=True
    )

    # 3.3 Position type diversity: at least one elected and at least one appointed among qualifying officials
    qualifying_types = [qualify_types[i] for i in qualifying_indices]
    has_elected = any(t == "elected" for t in qualifying_types)
    has_appointed = any(t == "appointed" for t in qualifying_types)
    type_diverse = has_elected and has_appointed
    evaluator.add_custom_node(
        result=type_diverse,
        id="Position_Type_Diversity_Elected_And_Appointed",
        desc="Verify that among the qualifying officials there is at least one elected federal official and at least one appointed federal official.",
        parent=constraints_node,
        critical=True
    )

    # Auxiliary info for debugging/analysis
    evaluator.add_custom_info(
        info={
            "qualifying_count": len(qualifying_indices),
            "qualifying_indices": qualifying_indices,
            "qualifying_states": list(qualifying_states_set),
            "has_elected": has_elected,
            "has_appointed": has_appointed,
        },
        info_type="set_level_summary"
    )

    # 4) Return structured summary
    return evaluator.get_summary()