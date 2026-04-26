import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "venezuela_feb12_2026_same_day"
TASK_DESCRIPTION = (
    "In February 2026, Venezuela's interim President Delcy Rodríguez gave her first interview to NBC's "
    "\"Meet the Press\" following the U.S. capture of Nicolás Maduro. On that same date, Venezuela's National Assembly "
    "took a specific action regarding the proposed amnesty bill for political prisoners. Identify: "
    "(1) the broadcast date of this NBC \"Meet the Press\" interview, "
    "(2) the specific action taken by the National Assembly on that date concerning the amnesty bill, and "
    "(3) provide reference URLs from credible news sources that verify both events occurred on the same day."
)

EXPECTED_TARGET_DATE_ISO = "2026-02-12"
EXPECTED_TARGET_DATE_TEXT = "February 12, 2026"
EXPECTED_ASSEMBLY_ACTION_KEYWORDS = ["postponed", "deferred", "delayed"]  # acceptable synonyms for "postponed"

# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #

_MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2, "febr": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def _strip_ordinal_suffix(s: str) -> str:
    return re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', s, flags=re.IGNORECASE)


def normalize_date_str(date_str: Optional[str]) -> Optional[str]:
    """
    Normalize common human-readable date strings to ISO YYYY-MM-DD.
    Supports forms like:
      - "February 12, 2026"
      - "Feb. 12, 2026"
      - "12 February 2026"
      - "2026-02-12" or "2026/02/12"
      - "Feb 12 2026"
      - "12 Feb 2026"
    Returns None if parsing fails.
    """
    if not date_str:
        return None

    s = date_str.strip()
    # Early ISO-like check
    m = re.match(r'^\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})\s*$', s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return f"{y:04d}-{mo:02d}-{d:02d}"
        except Exception:
            return None

    # Normalize punctuation and ordinals
    s = _strip_ordinal_suffix(s)
    s = s.replace(",", " ").replace(".", " ").replace("  ", " ").strip()
    tokens = s.split()
    lower_tokens = [t.lower() for t in tokens]

    # Patterns:
    # 1) Month Day Year
    if len(tokens) >= 3:
        # Find month position
        month_idx = None
        for idx, t in enumerate(lower_tokens):
            if t in _MONTH_MAP:
                month_idx = idx
                break
        if month_idx is not None:
            # If pattern Month Day Year e.g., ["February","12","2026"]
            if month_idx == 0 and len(tokens) >= 3:
                try:
                    mo = _MONTH_MAP[lower_tokens[0]]
                    d = int(tokens[1])
                    y = int(tokens[2])
                    return f"{y:04d}-{mo:02d}-{d:02d}"
                except Exception:
                    pass
            # If pattern Day Month Year e.g., ["12","February","2026"]
            if month_idx == 1 and len(tokens) >= 3:
                try:
                    d = int(tokens[0])
                    mo = _MONTH_MAP[lower_tokens[1]]
                    y = int(tokens[2])
                    return f"{y:04d}-{mo:02d}-{d:02d}"
                except Exception:
                    pass

    # Fallback: try to find a year and numbers around
    # Very rough heuristic: look for YYYY and two integers around it
    ym = re.search(r'(\d{4})', s)
    if ym:
        y = int(ym.group(1))
        # Find any month token near
        for name, mo in _MONTH_MAP.items():
            if name in s.lower():
                # Find first integer near it as day
                dm = re.search(r'\b(\d{1,2})\b', s)
                if dm:
                    try:
                        d = int(dm.group(1))
                        return f"{y:04d}-{mo:02d}-{d:02d}"
                    except Exception:
                        break
    return None


def clean_url_list(urls: Optional[List[str]]) -> List[str]:
    """Return a deduplicated list of non-empty, plausibly valid URLs."""
    if not urls:
        return []
    seen = set()
    res = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        # naive URL validity check
        if not re.match(r'^(https?://)', u):
            # If missing protocol, prepend http:// per framework recommendation
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            res.append(u)
    return res


def contains_action_postponed(text: Optional[str]) -> bool:
    """Heuristic check if an action string indicates 'postponed/deferred/delayed' debate."""
    if not text:
        return False
    lt = text.lower()
    return any(k in lt for k in EXPECTED_ASSEMBLY_ACTION_KEYWORDS) and ("debate" in lt or "discussion" in lt)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MainInfoExtraction(BaseModel):
    interview_date: Optional[str] = None
    interview_urls: List[str] = Field(default_factory=list)
    assembly_action: Optional[str] = None
    assembly_date: Optional[str] = None
    assembly_urls: List[str] = Field(default_factory=list)
    same_day_urls: List[str] = Field(default_factory=list)


class ConstraintsExtraction(BaseModel):
    foro_released_statement: Optional[str] = None
    foro_released_urls: List[str] = Field(default_factory=list)

    foro_remaining_statement: Optional[str] = None
    foro_remaining_urls: List[str] = Field(default_factory=list)

    two_votes_statement: Optional[str] = None
    two_votes_urls: List[str] = Field(default_factory=list)

    first_vote_statement: Optional[str] = None
    first_vote_urls: List[str] = Field(default_factory=list)

    jorge_release_statement: Optional[str] = None
    jorge_release_urls: List[str] = Field(default_factory=list)

    rescheduled_statement: Optional[str] = None
    rescheduled_urls: List[str] = Field(default_factory=list)

    foro_role_statement: Optional[str] = None
    foro_role_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_main_info() -> str:
    return """
Extract the following items exactly as they are presented in the answer text:

1) NBC "Meet the Press" interview:
   - interview_date: The broadcast date of Delcy Rodríguez's NBC "Meet the Press" interview.
   - interview_urls: A list of one or more URLs cited in the answer that support the interview occurrence and/or its date.

2) National Assembly action on the amnesty bill:
   - assembly_action: The specific action taken regarding the proposed amnesty bill (e.g., "postponed the debate").
   - assembly_date: The date on which that action occurred (as stated in the answer).
   - assembly_urls: A list of one or more URLs cited in the answer that support the Assembly action and its date.

3) Same-day verification:
   - same_day_urls: If the answer provides citation(s) intended to establish that both events occurred on the same date, extract those URLs here. If not explicitly provided, return an empty list.

Rules:
- Extract only what is explicitly present in the answer.
- Use the exact phrasing for dates and actions as written in the answer text.
- For URLs, extract all URLs relevant to the interview and assembly action; ignore irrelevant links.
- If a field is missing, set it to null (or empty list for URL fields).
"""


def prompt_extract_constraints() -> str:
    return """
Extract, as they appear in the answer, the statements (or their close paraphrases) and any URLs that support them for the following required conditions. If the answer does not include the statement, return null for the statement and an empty list for URLs.

1) foro_released_statement: A statement that, according to Foro Penal, more than 430 political prisoners (e.g., 431) had been confirmed released since January 8, 2026, as of February 12, 2026.
   - foro_released_urls: URLs cited for this.

2) foro_remaining_statement: A statement that, according to Foro Penal (as cited in Wikipedia per constraints), approximately 600 political prisoners remained detained as of February 12, 2026.
   - foro_remaining_urls: URLs cited for this.

3) two_votes_statement: A statement that the amnesty bill requires two votes to pass in the National Assembly.
   - two_votes_urls: URLs cited for this.

4) first_vote_statement: A statement that the first vote on the amnesty bill occurred on February 5, 2026, and passed unanimously.
   - first_vote_urls: URLs cited for this.

5) jorge_release_statement: A statement that National Assembly president Jorge Rodríguez said after the first vote that all concerned prisoners would be released by Friday, February 13, 2026.
   - jorge_release_urls: URLs cited for this.

6) rescheduled_statement: A statement that the debate postponed on February 12 was rescheduled to the following week.
   - rescheduled_urls: URLs cited for this.

7) foro_role_statement: A statement that Foro Penal is the NGO that tracks and verifies political prisoner releases in Venezuela.
   - foro_role_urls: URLs cited for this.

Rules:
- Extract the statements as they are written in the answer (or closely paraphrased if necessary).
- Extract only URLs that the answer associates with each specific statement.
- If a statement is missing, set the statement to null and the URL list to empty.
"""


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_interview_subtree(
    evaluator: Evaluator,
    parent: VerificationNode,
    main_info: MainInfoExtraction
) -> VerificationNode:
    """
    Build and verify the 'MeetThePress_Interview' subtree.
    Returns the parent subtree node.
    """
    interview_node = evaluator.add_parallel(
        id="MeetThePress_Interview",
        desc="Identify the broadcast date of Delcy Rodríguez's NBC 'Meet the Press' interview with credible sourcing",
        parent=parent,
        critical=True
    )

    # Leaf: Interview date stated as Feb 12, 2026
    leaf_date_stated = evaluator.add_leaf(
        id="Interview_Date_Is_Feb12_2026",
        desc="States that the interview broadcast date is February 12, 2026",
        parent=interview_node,
        critical=True
    )
    claim_date_stated = (
        "The answer explicitly states that the broadcast date of Delcy Rodríguez's NBC 'Meet the Press' interview "
        f"is {EXPECTED_TARGET_DATE_TEXT}."
    )
    await evaluator.verify(
        claim=claim_date_stated,
        node=leaf_date_stated,
        additional_instruction=(
            "Check the answer text for a statement that the broadcast date is exactly February 12, 2026 "
            "(allow minor formatting variations like 'Feb. 12, 2026' or 'Feb 12, 2026'). "
            "You only need to verify that the answer asserts this date."
        )
    )

    # Leaf: Interview source URL provided
    interview_urls = clean_url_list(main_info.interview_urls)
    evaluator.add_custom_node(
        result=len(interview_urls) > 0,
        id="Interview_Source_URL_Provided",
        desc="Provides at least one credible reference URL supporting the interview and its broadcast date",
        parent=interview_node,
        critical=True
    )

    # Leaf: Interview source supports Feb 12, 2026
    leaf_source_supports = evaluator.add_leaf(
        id="Interview_Source_Supports_Feb12_2026",
        desc="The provided interview source URL supports that the broadcast date is February 12, 2026",
        parent=interview_node,
        critical=True
    )
    claim_source_supports = (
        "The NBC 'Meet the Press' interview with Delcy Rodríguez was broadcast on February 12, 2026."
    )
    # If URLs exist, verify against them; otherwise still run a simple verification (parent will fail if URL missing)
    await evaluator.verify(
        claim=claim_source_supports,
        node=leaf_source_supports,
        sources=interview_urls if interview_urls else None,
        additional_instruction=(
            "Verify that the cited page(s) explicitly confirm the interview's broadcast date as February 12, 2026. "
            "Allow minor formatting variations in the date string."
        )
    )

    return interview_node


async def build_assembly_subtree(
    evaluator: Evaluator,
    parent: VerificationNode,
    main_info: MainInfoExtraction
) -> VerificationNode:
    """
    Build and verify the 'NationalAssembly_Amnesty_Action' subtree.
    Returns the parent subtree node.
    """
    assembly_node = evaluator.add_parallel(
        id="NationalAssembly_Amnesty_Action",
        desc="Identify the National Assembly action regarding the proposed amnesty bill on that date with credible sourcing",
        parent=parent,
        critical=True
    )

    # Leaf: Assembly action is postponed debate
    leaf_action = evaluator.add_leaf(
        id="Assembly_Action_Is_Postponed_Debate",
        desc="States that the National Assembly postponed the debate on the proposed amnesty bill",
        parent=assembly_node,
        critical=True
    )
    claim_action_stated = (
        "The answer states that the Venezuelan National Assembly postponed the debate on the proposed amnesty bill "
        "(accept equivalent phrasings like 'deferred the debate' or 'delayed discussion')."
    )
    await evaluator.verify(
        claim=claim_action_stated,
        node=leaf_action,
        additional_instruction=(
            "Focus only on whether the answer asserts that the debate on the proposed amnesty bill was postponed "
            "(or clearly synonymous phrasing such as deferred/delayed). Do not check external sources here."
        )
    )

    # Leaf: Assembly action date is Feb 12, 2026
    leaf_action_date = evaluator.add_leaf(
        id="Assembly_Action_Date_Is_Feb12_2026",
        desc="States that this postponement occurred on February 12, 2026",
        parent=assembly_node,
        critical=True
    )
    claim_action_date_stated = (
        "The answer explicitly states that the postponement of the amnesty bill debate occurred on February 12, 2026."
    )
    await evaluator.verify(
        claim=claim_action_date_stated,
        node=leaf_action_date,
        additional_instruction=(
            "Check the answer text for a statement that this action occurred on February 12, 2026 "
            "(allow minor formatting variations like 'Feb. 12, 2026' or 'Feb 12, 2026')."
        )
    )

    # Leaf: Assembly source URL provided
    assembly_urls = clean_url_list(main_info.assembly_urls)
    evaluator.add_custom_node(
        result=len(assembly_urls) > 0,
        id="Assembly_Source_URL_Provided",
        desc="Provides at least one credible reference URL supporting the Assembly action and its date",
        parent=assembly_node,
        critical=True
    )

    # Leaf: Assembly source supports both action and date
    leaf_source_supports_action_date = evaluator.add_leaf(
        id="Assembly_Source_Supports_Action_And_Date",
        desc="The provided Assembly source URL supports both (a) postponement of the debate and (b) the date February 12, 2026",
        parent=assembly_node,
        critical=True
    )
    claim_action_date_supported = (
        "The Venezuelan National Assembly postponed the debate on the proposed amnesty bill on February 12, 2026."
    )
    await evaluator.verify(
        claim=claim_action_date_supported,
        node=leaf_source_supports_action_date,
        sources=assembly_urls if assembly_urls else None,
        additional_instruction=(
            "Verify that the cited page(s) explicitly confirm both items: "
            "(1) the debate on the proposed amnesty bill was postponed (or deferred/delayed), and "
            f"(2) the date of that postponement is {EXPECTED_TARGET_DATE_TEXT}."
        )
    )

    return assembly_node


def build_same_day_subtree(
    evaluator: Evaluator,
    parent: VerificationNode,
    main_info: MainInfoExtraction,
    interview_node: VerificationNode,
    assembly_node: VerificationNode
) -> VerificationNode:
    """
    Build the 'Same_Day_Verification' subtree. Uses custom checks for date equality
    and derives a conservative 'citations support same day' judgment based on prior leaves.
    Returns the parent subtree node.
    """
    same_day_node = evaluator.add_parallel(
        id="Same_Day_Verification",
        desc="Provide citations showing both events occurred on the same date",
        parent=parent,
        critical=True
    )

    # Leaf: Dates match (both are Feb 12, 2026)
    norm_interview = normalize_date_str(main_info.interview_date)
    norm_assembly = normalize_date_str(main_info.assembly_date)
    dates_match = (norm_interview == EXPECTED_TARGET_DATE_ISO) and (norm_assembly == EXPECTED_TARGET_DATE_ISO)
    evaluator.add_custom_node(
        result=dates_match,
        id="Dates_Match_Same_Day",
        desc="The stated interview date and the stated Assembly-action date are the same calendar date (Feb 12, 2026)",
        parent=same_day_node,
        critical=True
    )

    # Leaf: Citations support same day (approximated by earlier URL-backed verifications passing)
    # We conservatively require that the prior URL-backed leaves have passed.
    # Find those child leaves under the respective subtrees
    # Note: We'll search for leaves by ID.
    interview_support = evaluator.find_node("Interview_Source_Supports_Feb12_2026")
    assembly_support = evaluator.find_node("Assembly_Source_Supports_Action_And_Date")

    citations_support = (interview_support is not None and interview_support.status == "passed") and \
                        (assembly_support is not None and assembly_support.status == "passed")

    evaluator.add_custom_node(
        result=citations_support,
        id="Citations_Support_Same_Day",
        desc="The provided citations (one or more URLs) jointly support that both events occurred on that same date",
        parent=same_day_node,
        critical=True
    )

    return same_day_node


async def build_constraints_subtree(
    evaluator: Evaluator,
    parent: VerificationNode,
    constraints: ConstraintsExtraction
) -> VerificationNode:
    """
    Build and verify the 'Additional_Constraints_From_List' subtree.
    Uses simple verifications focusing on whether the answer includes each required statement.
    """
    constraints_node = evaluator.add_parallel(
        id="Additional_Constraints_From_List",
        desc="All additional facts listed in the constraints are included as required conditions",
        parent=parent,
        critical=True
    )

    # 1) Foro Penal releases count as of Feb 12
    leaf_foro_released = evaluator.add_leaf(
        id="ForoPenal_Releases_Count_AsOf_Feb12",
        desc="States that, according to Foro Penal, more than 430 political prisoners (431 per the constraint text) had been confirmed released since January 8, 2026, as of February 12, 2026",
        parent=constraints_node,
        critical=True
    )
    claim_foro_released = (
        "The answer states that, according to Foro Penal, more than 430 political prisoners "
        "(e.g., 431) had been confirmed released since January 8, 2026, as of February 12, 2026."
    )
    await evaluator.verify(
        claim=claim_foro_released,
        node=leaf_foro_released,
        additional_instruction=(
            "Only check whether the answer explicitly includes this statement (or an equivalent paraphrase). "
            "Do not verify external truth here."
        )
    )

    # 2) Foro Penal remaining detained as of Feb 12
    leaf_foro_remaining = evaluator.add_leaf(
        id="ForoPenal_Remaining_Detained_AsOf_Feb12",
        desc="States that, according to Foro Penal (as cited in Wikipedia per constraints), approximately 600 political prisoners remained detained as of February 12, 2026",
        parent=constraints_node,
        critical=True
    )
    claim_foro_remaining = (
        "The answer states that, according to Foro Penal (as cited in Wikipedia per constraints), "
        "approximately 600 political prisoners remained detained as of February 12, 2026."
    )
    await evaluator.verify(
        claim=claim_foro_remaining,
        node=leaf_foro_remaining,
        additional_instruction=(
            "Focus on whether the answer asserts this item (allow words like 'about', 'around', or '~600')."
        )
    )

    # 3) Amnesty bill requires two votes
    leaf_two_votes = evaluator.add_leaf(
        id="Amnesty_Bill_Two_Votes_Required",
        desc="States that the amnesty bill requires two votes to pass in the National Assembly",
        parent=constraints_node,
        critical=True
    )
    claim_two_votes = (
        "The answer states that the amnesty bill requires two votes to pass in the National Assembly."
    )
    await evaluator.verify(
        claim=claim_two_votes,
        node=leaf_two_votes,
        additional_instruction="Only check if the answer asserts this requirement."
    )

    # 4) First vote Feb 5 unanimous
    leaf_first_vote = evaluator.add_leaf(
        id="First_Vote_Feb5_Unanimous",
        desc="States that the first vote on the amnesty bill occurred on February 5, 2026, and passed unanimously",
        parent=constraints_node,
        critical=True
    )
    claim_first_vote = (
        "The answer states that the first vote on the amnesty bill occurred on February 5, 2026, and passed unanimously."
    )
    await evaluator.verify(
        claim=claim_first_vote,
        node=leaf_first_vote,
        additional_instruction=(
            "Check that both the date (February 5, 2026) and the 'unanimous' outcome are present in the answer."
        )
    )

    # 5) Jorge Rodríguez statement about releases by Friday, Feb 13, 2026
    leaf_jorge = evaluator.add_leaf(
        id="JorgeRodriguez_ReleaseBy_Feb13_Statement",
        desc="States that National Assembly president Jorge Rodríguez said after the first vote that all concerned prisoners would be released by Friday, February 13, 2026",
        parent=constraints_node,
        critical=True
    )
    claim_jorge = (
        "The answer states that National Assembly president Jorge Rodríguez said, after the first vote, "
        "that all concerned prisoners would be released by Friday, February 13, 2026."
    )
    await evaluator.verify(
        claim=claim_jorge,
        node=leaf_jorge,
        additional_instruction=(
            "Allow minor phrasing variations (e.g., 'by Friday 13 February 2026'). Focus on whether the answer asserts this."
        )
    )

    # 6) Postponed debate rescheduled to following week
    leaf_rescheduled = evaluator.add_leaf(
        id="Postponed_Debate_Rescheduled_Following_Week",
        desc="States that the debate postponed on February 12 was rescheduled to the following week",
        parent=constraints_node,
        critical=True
    )
    claim_rescheduled = (
        "The answer states that the debate postponed on February 12 was rescheduled to the following week."
    )
    await evaluator.verify(
        claim=claim_rescheduled,
        node=leaf_rescheduled,
        additional_instruction="Check only whether the answer asserts this scheduling detail."
    )

    # 7) Foro Penal role
    leaf_foro_role = evaluator.add_leaf(
        id="ForoPenal_Role_Tracker",
        desc="States that Foro Penal is the NGO that tracks and verifies political prisoner releases in Venezuela",
        parent=constraints_node,
        critical=True
    )
    claim_foro_role = (
        "The answer states that Foro Penal is the NGO that tracks and verifies political prisoner releases in Venezuela."
    )
    await evaluator.verify(
        claim=claim_foro_role,
        node=leaf_foro_role,
        additional_instruction="Check only whether the answer asserts this role."
    )

    return constraints_node


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
    Evaluate an answer for the Venezuela same-day events task (Feb 12, 2026).
    """
    # Initialize evaluator
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

    # Add a task-level critical node mirroring the rubric "Root"
    task_root = evaluator.add_parallel(
        id="Root",
        desc="Answer satisfies the proposed question and all listed constraints, with credible sourcing where required",
        parent=root,
        critical=True
    )

    # Run extractions
    main_info, constraints_info = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_main_info(),
            template_class=MainInfoExtraction,
            extraction_name="main_info",
        ),
        evaluator.extract(
            prompt=prompt_extract_constraints(),
            template_class=ConstraintsExtraction,
            extraction_name="constraints_info",
        ),
    )

    # Record a small ground-truth expectation for transparency
    evaluator.add_ground_truth(
        {
            "expected_date": EXPECTED_TARGET_DATE_TEXT,
            "expected_assembly_action_keywords": EXPECTED_ASSEMBLY_ACTION_KEYWORDS
        },
        gt_type="expected_targets"
    )

    # Build Required_Identifications subtree
    required_node = evaluator.add_parallel(
        id="Required_Identifications",
        desc="Provide the three requested outputs (interview date, Assembly action, and same-day verification via citations)",
        parent=task_root,
        critical=True
    )

    # Subtree: Interview
    interview_parent = await build_interview_subtree(evaluator, required_node, main_info)

    # Subtree: Assembly
    assembly_parent = await build_assembly_subtree(evaluator, required_node, main_info)

    # Subtree: Same Day Verification
    build_same_day_subtree(
        evaluator=evaluator,
        parent=required_node,
        main_info=main_info,
        interview_node=interview_parent,
        assembly_node=assembly_parent
    )

    # Build Additional Constraints subtree
    await build_constraints_subtree(evaluator, task_root, constraints_info)

    # Return structured summary
    return evaluator.get_summary()