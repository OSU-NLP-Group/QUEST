import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "state_veto_override_2024_2025"
TASK_DESCRIPTION = """
Identify a state legislative veto override that occurred between January 1, 2024, and December 31, 2025 (inclusive), where ALL of the following conditions are met:

1. The state's constitution or statutes require a three-fifths (3/5) supermajority vote in both legislative chambers to override a gubernatorial veto.

2. The override occurred during a regular legislative session (not a special or extraordinary session).

3. The bill that was vetoed and subsequently overridden relates to property tax reform, healthcare policy, or civil rights protections.

4. The state legislature meets in regular session every calendar year (not just in odd-numbered years like Montana, Nevada, North Dakota, and Texas).

5. The governor who issued the veto belonged to the same political party that held the majority in both legislative chambers at the time of the override vote.

6. The bill was originally introduced in the state house of representatives (or equivalent lower chamber) rather than the state senate.

7. The override vote in at least one chamber exceeded the exact three-fifths requirement (received more votes than the minimum needed).

Provide the following information:
- The name of the state
- The bill number and title
- The date of the veto override
- The vote tallies in both chambers
- The policy area of the bill
- Reference URLs supporting each requirement
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConstraintURLs(BaseModel):
    timeframe: List[str] = Field(default_factory=list)
    three_fifths_rule: List[str] = Field(default_factory=list)
    regular_session: List[str] = Field(default_factory=list)
    policy_area: List[str] = Field(default_factory=list)
    annual_sessions: List[str] = Field(default_factory=list)
    party_match: List[str] = Field(default_factory=list)
    house_origination: List[str] = Field(default_factory=list)
    exceeds_three_fifths: List[str] = Field(default_factory=list)
    event_details: List[str] = Field(default_factory=list)


class OverrideEventExtraction(BaseModel):
    state: Optional[str] = None
    bill_number: Optional[str] = None
    bill_title: Optional[str] = None
    override_date: Optional[str] = None  # Prefer ISO or unambiguous date string
    house_vote_tally: Optional[str] = None  # e.g., "65-28"
    senate_vote_tally: Optional[str] = None  # e.g., "24-8"
    policy_area: Optional[str] = None  # One of: property tax reform, healthcare policy, civil rights protections
    origination_chamber: Optional[str] = None  # e.g., "house", "lower chamber", "senate"
    session_type: Optional[str] = None  # e.g., "regular", "special", "extraordinary"
    governor_party: Optional[str] = None
    house_majority_party: Optional[str] = None
    senate_majority_party: Optional[str] = None
    override_requirement_summary: Optional[str] = None  # e.g., "3/5 in both chambers"
    legislature_meets_annually_claim: Optional[str] = None  # e.g., "Yes, annual regular sessions"
    urls: ConstraintURLs = Field(default_factory=ConstraintURLs)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_override_event() -> str:
    return """
    Extract information about a single state legislative veto override event that the answer claims satisfies the specified constraints. If multiple events are mentioned, extract ONLY the strongest/first one that appears to satisfy the constraints. Do not invent information beyond what is stated.

    Required fields:
    - state: The state name.
    - bill_number: The bill number (e.g., "HB 68").
    - bill_title: The bill title.
    - override_date: The date of the veto override (use an unambiguous date string; ISO format preferred if available).
    - house_vote_tally: The vote tally in the lower chamber (e.g., "65-28"). If not provided, set to null.
    - senate_vote_tally: The vote tally in the upper chamber (e.g., "24-8"). If not provided, set to null.
    - policy_area: One of: "property tax reform", "healthcare policy", or "civil rights protections". If the answer uses a close synonym, choose the best-matching one. If unclear, set to null.
    - origination_chamber: The origination chamber of the bill (e.g., "house" / "lower chamber" / "senate"). If unclear, set to null.
    - session_type: The session type during the override (e.g., "regular", "special", "extraordinary") if stated or inferable from the cited description in the answer; otherwise null.
    - governor_party: The governor's party at the time of the override (e.g., "Republican", "Democrat"), if provided; else null.
    - house_majority_party: The party holding majority in the lower chamber at the time, if provided; else null.
    - senate_majority_party: The party holding majority in the upper chamber at the time, if provided; else null.
    - override_requirement_summary: A short phrase summarizing the override threshold if explicitly mentioned (e.g., "3/5 in both chambers"); else null.
    - legislature_meets_annually_claim: A short phrase capturing whether the state meets in regular session annually (e.g., "annual regular sessions"); else null.

    Reference URLs:
    Extract URLs that the answer explicitly provides to support EACH of the following:
    - urls.timeframe: URLs supporting the actual date/timeframe of the override (2024–2025 inclusive).
    - urls.three_fifths_rule: URLs supporting that the state's constitution/statutes require a 3/5 vote in BOTH chambers to override a governor's veto.
    - urls.regular_session: URLs supporting that the override occurred during a REGULAR session (not special/extraordinary).
    - urls.policy_area: URLs supporting the bill's policy area classification (property tax reform, healthcare policy, or civil rights protections).
    - urls.annual_sessions: URLs supporting that the legislature meets in regular session EVERY calendar year.
    - urls.party_match: URLs supporting that the governor's party matched the majority party in BOTH chambers at the time of the override.
    - urls.house_origination: URLs supporting that the bill originated in the lower chamber (house) rather than the senate.
    - urls.exceeds_three_fifths: URLs supporting that AT LEAST ONE chamber's override vote exceeded the exact 3/5 minimum needed.
    - urls.event_details: General URLs about the event (bill page, journals, news) that include date, chamber votes, bill number/title, etc. If none are provided, leave empty.

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer text (including markdown links). Do NOT infer or invent URLs.
    - If a particular URL category is not provided in the answer, return an empty list for that category.
    - Return null for any scalar field that is not explicitly stated in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_event_constraints_subtree(evaluator: Evaluator, parent_node, data: OverrideEventExtraction):
    """
    Build and verify the 'Event_Meets_All_Constraints' subtree (all critical leaves).
    """
    constraints_node = evaluator.add_parallel(
        id="Event_Meets_All_Constraints",
        desc="The identified veto override event satisfies all stated constraints.",
        parent=parent_node,
        critical=True
    )

    # 1) Override_Timeframe
    timeframe_node = evaluator.add_leaf(
        id="Override_Timeframe",
        desc="The veto override occurred between January 1, 2024, and December 31, 2025 (inclusive).",
        parent=constraints_node,
        critical=True
    )
    tf_claim_date = (
        f"The veto override occurred on {data.override_date}, which falls between Jan 1, 2024 and Dec 31, 2025 (inclusive)."
        if _nonempty(data.override_date)
        else "The veto override occurred between Jan 1, 2024 and Dec 31, 2025 (inclusive)."
    )
    await evaluator.verify(
        claim=tf_claim_date,
        node=timeframe_node,
        sources=data.urls.timeframe,
        additional_instruction="Verify the actual override date on the provided page(s). If the exact date matches and is within 2024–2025 inclusive, pass."
    )

    # 2) Three_Fifths_Override_Requirement
    req35_node = evaluator.add_leaf(
        id="Three_Fifths_Override_Requirement",
        desc="The state's constitution or statutes require a three-fifths (3/5) supermajority vote in BOTH legislative chambers to override a gubernatorial veto.",
        parent=constraints_node,
        critical=True
    )
    state_txt = data.state or "the state"
    claim_35 = f"In {state_txt}, overriding a gubernatorial veto requires a three‑fifths (3/5) supermajority vote in BOTH legislative chambers."
    await evaluator.verify(
        claim=claim_35,
        node=req35_node,
        sources=data.urls.three_fifths_rule,
        additional_instruction="Confirm the requirement is explicitly 3/5 and applies to BOTH chambers (not only one). Primary sources like constitution/statutes preferred."
    )

    # 3) Regular_Session_Override
    regular_session_node = evaluator.add_leaf(
        id="Regular_Session_Override",
        desc="The override occurred during a regular legislative session (not a special or extraordinary session).",
        parent=constraints_node,
        critical=True
    )
    reg_claim = (
        f"The veto override occurred during a regular legislative session (not special/extraordinary)."
    )
    await evaluator.verify(
        claim=reg_claim,
        node=regular_session_node,
        sources=data.urls.regular_session,
        additional_instruction="Pass if the source explicitly indicates 'regular session' or clearly places the action in the state's regular session (e.g., bill page or journal labeled with the regular session). Do not pass if the source indicates 'special' or 'extraordinary' session."
    )

    # 4) Policy_Area_Match
    policy_match_node = evaluator.add_leaf(
        id="Policy_Area_Match",
        desc="The vetoed/overridden bill relates to property tax reform, healthcare policy, or civil rights protections.",
        parent=constraints_node,
        critical=True
    )
    policy_area_txt = data.policy_area or "one of: property tax reform, healthcare policy, or civil rights protections"
    pol_claim = f"The bill addresses {policy_area_txt}."
    await evaluator.verify(
        claim=pol_claim,
        node=policy_match_node,
        sources=data.urls.policy_area,
        additional_instruction="Verify the policy area stated on the page; map close synonyms (e.g., LGBTQ+ rights → civil rights protections; medical care restrictions → healthcare policy)."
    )

    # 5) Annual_Legislative_Sessions
    annual_node = evaluator.add_leaf(
        id="Annual_Legislative_Sessions",
        desc="The state legislature meets in regular session every calendar year (not only in odd-numbered years).",
        parent=constraints_node,
        critical=True
    )
    annual_claim = f"{state_txt} legislature meets in regular session every calendar year (i.e., annual regular sessions)."
    await evaluator.verify(
        claim=annual_claim,
        node=annual_node,
        sources=data.urls.annual_sessions,
        additional_instruction="Accept authoritative references (constitution/statutes/official legislative site/NCSL) that indicate annual regular sessions."
    )

    # 6) Governor_Party_Matches_Legislative_Majorities
    party_match_node = evaluator.add_leaf(
        id="Governor_Party_Matches_Legislative_Majorities",
        desc="At the time of the override vote, the governor who issued the veto belonged to the same political party that held the majority in BOTH legislative chambers.",
        parent=constraints_node,
        critical=True
    )
    # Build a general claim; specific party names are optional
    if _nonempty(data.governor_party) and _nonempty(data.house_majority_party) and _nonempty(data.senate_majority_party):
        pm_claim = (
            f"At the time of the override, the governor was {data.governor_party}, and both chambers had {data.house_majority_party} (house) "
            f"and {data.senate_majority_party} (senate) majorities—i.e., the governor's party matched the majority party in BOTH chambers."
        )
    else:
        pm_claim = "At the time of the override, the governor's party matched the majority party in BOTH legislative chambers."

    await evaluator.verify(
        claim=pm_claim,
        node=party_match_node,
        sources=data.urls.party_match,
        additional_instruction="Pass if the page indicates the governor and both chambers were controlled by the same party (e.g., 'Republican-led legislature overrode Republican governor's veto')."
    )

    # 7) House_Origination
    house_orig_node = evaluator.add_leaf(
        id="House_Origination",
        desc="The bill was originally introduced in the state house of representatives (or equivalent lower chamber), not the state senate.",
        parent=constraints_node,
        critical=True
    )
    bill_ref = f"{data.bill_number}" if _nonempty(data.bill_number) else "the bill"
    ho_claim = f"{bill_ref} originated in the state's lower chamber (house), not the senate."
    await evaluator.verify(
        claim=ho_claim,
        node=house_orig_node,
        sources=data.urls.house_origination,
        additional_instruction="Accept as evidence: official bill pages stating 'House Bill', 'Introduced in House', or similar clear indicators that origination was in the lower chamber."
    )

    # 8) Vote_Margin_Exceeded_In_At_Least_One_Chamber
    exceed_node = evaluator.add_leaf(
        id="Vote_Margin_Exceeded_In_At_Least_One_Chamber",
        desc="In at least one chamber, the override vote exceeded (was greater than) the exact three-fifths minimum required.",
        parent=constraints_node,
        critical=True
    )
    ex_claim = (
        "At least one chamber's override vote exceeded (was greater than) the exact three‑fifths minimum required for a veto override."
    )
    await evaluator.verify(
        claim=ex_claim,
        node=exceed_node,
        sources=data.urls.exceeds_three_fifths,
        additional_instruction="Pass if the page explicitly states the vote exceeded the 3/5 threshold OR if the page shows a tally clearly greater than 60% of the chamber's voting membership/required votes for override."
    )


async def build_required_fields_subtree(evaluator: Evaluator, parent_node, data: OverrideEventExtraction):
    """
    Build and verify the 'Response_Provides_Required_Fields' subtree:
    - Presence checks for required fields (custom boolean leaves)
    - URL support checks for each constraint (via verification by URLs)
    """
    fields_node = evaluator.add_parallel(
        id="Response_Provides_Required_Fields",
        desc="The response includes all requested output fields for the identified event.",
        parent=parent_node,
        critical=True
    )

    # Presence checks (critical)
    evaluator.add_custom_node(
        result=_nonempty(data.state),
        id="State_Name_Provided",
        desc="Provides the name of the state.",
        parent=fields_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(data.bill_number),
        id="Bill_Number_Provided",
        desc="Provides the bill number.",
        parent=fields_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(data.bill_title),
        id="Bill_Title_Provided",
        desc="Provides the bill title.",
        parent=fields_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(data.override_date),
        id="Override_Date_Provided",
        desc="Provides the date of the veto override.",
        parent=fields_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(data.house_vote_tally) and _nonempty(data.senate_vote_tally),
        id="Vote_Tallies_Both_Chambers_Provided",
        desc="Provides the vote tallies in both chambers.",
        parent=fields_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(data.policy_area),
        id="Policy_Area_Provided",
        desc="States the policy area of the bill (property tax reform, healthcare policy, or civil rights protections).",
        parent=fields_node,
        critical=True
    )

    # URL support subtree (critical parallel)
    url_support_node = evaluator.add_parallel(
        id="Reference_URLs_For_Each_Constraint",
        desc="Provides reference URL(s) supporting each stated constraint.",
        parent=fields_node,
        critical=True
    )

    # Helper to add URL-support verification leaves
    async def _add_url_support_leaf(node_id: str, description: str, claim: str, urls: List[str], add_ins: str):
        node = evaluator.add_leaf(
            id=node_id,
            desc=description,
            parent=url_support_node,
            critical=True
        )
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=add_ins
        )

    # 1) Timeframe URL support
    claim_timeframe = (
        f"The veto override occurred on {data.override_date}, which is within Jan 1, 2024–Dec 31, 2025 (inclusive)."
        if _nonempty(data.override_date)
        else "The veto override occurred within Jan 1, 2024–Dec 31, 2025 (inclusive)."
    )
    await _add_url_support_leaf(
        "URL_Supports_Timeframe",
        "Provides at least one reference URL supporting that the override occurred within Jan 1, 2024–Dec 31, 2025 (inclusive).",
        claim_timeframe,
        data.urls.timeframe,
        "Verify the override date on the page and confirm it falls within 2024–2025 inclusive."
    )

    # 2) 3/5 rule URL support
    await _add_url_support_leaf(
        "URL_Supports_Three_Fifths_Rule",
        "Provides at least one reference URL supporting the 3/5 override requirement in both chambers (constitution/statute).",
        f"In {data.state or 'the state'}, overriding a governor's veto requires a three‑fifths vote in BOTH chambers.",
        data.urls.three_fifths_rule,
        "Pass only if the page clearly states a 3/5 threshold applies to both chambers."
    )

    # 3) Regular session URL support
    await _add_url_support_leaf(
        "URL_Supports_Regular_Session",
        "Provides at least one reference URL supporting that the override occurred during a regular session (not special/extraordinary).",
        "The veto override occurred during a regular legislative session (not a special or extraordinary session).",
        data.urls.regular_session,
        "Look for explicit 'regular session' labeling or clear placement in the state's normal session; do not pass if the page indicates special/extraordinary."
    )

    # 4) Policy area URL support
    await _add_url_support_leaf(
        "URL_Supports_Policy_Area",
        "Provides at least one reference URL supporting the bill's policy area match (property tax reform OR healthcare policy OR civil rights protections).",
        f"The bill concerns {data.policy_area or 'one of: property tax reform, healthcare policy, or civil rights protections'}.",
        data.urls.policy_area,
        "Map close synonyms to the allowed categories (e.g., gender‑affirming care policy → healthcare policy; anti‑discrimination protections → civil rights protections)."
    )

    # 5) Annual sessions URL support
    await _add_url_support_leaf(
        "URL_Supports_Annual_Sessions",
        "Provides at least one reference URL supporting that the legislature meets in regular session every calendar year.",
        f"{data.state or 'The state'} legislature meets in regular session every calendar year.",
        data.urls.annual_sessions,
        "Authoritative sources preferred (constitution, official legislature site, NCSL)."
    )

    # 6) Party match URL support
    await _add_url_support_leaf(
        "URL_Supports_Party_Match",
        "Provides at least one reference URL supporting that the governor's party matched the majority party in both chambers at the time of the override.",
        "At the time of the override, the governor's party matched the majority party in BOTH legislative chambers.",
        data.urls.party_match,
        "Accept if the page indicates the legislature was led by the same party as the governor (e.g., 'Republican‑led House and Senate overrode Republican governor's veto')."
    )

    # 7) House origination URL support
    await _add_url_support_leaf(
        "URL_Supports_House_Origination",
        "Provides at least one reference URL supporting that the bill originated in the lower chamber (house) rather than the senate.",
        f"{data.bill_number or 'The bill'} originated in the state's lower chamber (house), not the senate.",
        data.urls.house_origination,
        "Bill pages that say 'House Bill' or 'Introduced in House' count as support."
    )

    # 8) Exceeds 3/5 URL support
    await _add_url_support_leaf(
        "URL_Supports_Exceeds_Three_Fifths",
        "Provides at least one reference URL supporting that at least one chamber's override vote exceeded the exact 3/5 minimum needed.",
        "At least one chamber's override vote exceeded the exact three‑fifths minimum required for a veto override.",
        data.urls.exceeds_three_fifths,
        "Pass if the page explicitly says the vote exceeded the 3/5 threshold OR clearly shows a tally greater than 60% of the chamber's required votes."
    )


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the state legislative veto override task (2024–2025).
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_override_event(),
        template_class=OverrideEventExtraction,
        extraction_name="override_event_extraction",
    )

    # Build rubric root (critical parallel)
    rubric_root = evaluator.add_parallel(
        id="Legislative_Veto_Override_Identification",
        desc="Evaluates whether the response identifies a single qualifying state legislative veto override event (2024-2025) and provides all requested fields with supporting URLs.",
        parent=root,
        critical=True
    )

    # Subtree: Event constraints (critical, parallel)
    await build_event_constraints_subtree(evaluator, rubric_root, extracted)

    # Subtree: Required fields and URL support (critical, parallel)
    await build_required_fields_subtree(evaluator, rubric_root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()