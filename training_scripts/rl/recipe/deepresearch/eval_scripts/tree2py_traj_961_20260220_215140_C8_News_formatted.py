import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "trump_cabinet_2025_confirmations"
TASK_DESCRIPTION = (
    "Research the Trump administration's second-term Cabinet confirmation process during January-February 2025 to analyze Senate confirmation patterns and voting behavior. "
    "Identify the three Cabinet officials confirmed earliest in chronological order during this period. For each of these three officials, provide: "
    "(1) their Cabinet position title, (2) their name, (3) the exact date of Senate confirmation, (4) the final Senate vote count, and (5) whether a Vice Presidential tie-breaking vote was required. "
    "Additionally, identify the House Speaker elected at the beginning of the 119th Congress in January 2025, providing: (1) the Speaker's name, (2) the election date, and (3) the final vote count."
)

ROOT_DESC = "Comprehensive analysis of Trump administration's second-term Cabinet confirmation process and congressional leadership during January-February 2025"

# Expected reference values per rubric (used for targeted checks)
CABINET_EXPECTATIONS = [
    {
        "group_id": "first",
        "group_desc": "First Cabinet official confirmed (earliest chronological confirmation date in 2025)",
        "expected_position": ["Secretary of State"],
        "expected_date": "January 20, 2025",
        "expected_vote": "99-0",
        "expected_tiebreaker": "no",
    },
    {
        "group_id": "second",
        "group_desc": "Second Cabinet official confirmed (second-earliest chronological confirmation date in 2025)",
        "expected_position": ["Secretary of Defense", "Secretary of War"],
        "expected_date": "January 24, 2025",
        "expected_vote": "51-50",
        "expected_tiebreaker": "yes",
    },
    {
        "group_id": "third",
        "group_desc": "Third Cabinet official confirmed (third-earliest chronological confirmation date in 2025)",
        "expected_position": ["Attorney General"],
        "expected_date": "February 4, 2025",
        "expected_vote": "54-46",
        "expected_tiebreaker": "no",
    },
]

SPEAKER_EXPECTATIONS = {
    "group_id": "speaker",
    "group_desc": "House Speaker elected at the beginning of the 119th Congress in January 2025",
    "expected_date": "January 3, 2025",
    "expected_vote": "218-215",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CabinetOfficial(BaseModel):
    position_title: Optional[str] = None
    name: Optional[str] = None
    confirmation_date: Optional[str] = None
    final_vote_count: Optional[str] = None  # e.g., "99-0", "54-46"
    tiebreaker_required: Optional[str] = None  # e.g., "yes", "no", "required", "not required"
    vote_nature: Optional[str] = None  # e.g., "unanimous", "narrow majority", "tie"
    sources: List[str] = Field(default_factory=list)


class CabinetExtraction(BaseModel):
    first: Optional[CabinetOfficial] = None
    second: Optional[CabinetOfficial] = None
    third: Optional[CabinetOfficial] = None


class SpeakerInfo(BaseModel):
    name: Optional[str] = None
    election_date: Optional[str] = None
    final_vote_count: Optional[str] = None
    vote_details: Optional[str] = None  # e.g., "decided on first ballot despite initial holdouts"
    sources: List[str] = Field(default_factory=list)


class FullExtraction(BaseModel):
    cabinet: Optional[CabinetExtraction] = None
    speaker: Optional[SpeakerInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_cabinet() -> str:
    return """
    Extract from the answer the three Cabinet officials confirmed earliest in chronological order during January–February 2025.
    Return a JSON object with a top-level key `cabinet` containing objects `first`, `second`, and `third`.
    For each of `first`, `second`, and `third`, extract the following fields exactly as stated in the answer:
    - position_title: the Cabinet position title (string)
    - name: the official’s full name (string)
    - confirmation_date: the exact Senate confirmation date as presented (string)
    - final_vote_count: the final Senate vote count (string, e.g., "99-0", "51-50", "54-46")
    - tiebreaker_required: whether a Vice Presidential tie-breaking vote was required (string; e.g., "yes", "no", "required", "not required")
    - vote_nature: any textual characterization of the vote (string; e.g., "unanimous", "narrow majority", "partisan")
    - sources: a list of all URLs cited in the answer for this official (including Senate.gov, Congress.gov, or news pages). Extract only actual URLs present in the answer.

    If any field is missing for an item, set it to null. If there are no URLs for an item, return an empty list for `sources`.
    Do not invent any information not explicitly present in the answer.
    """


def prompt_extract_speaker() -> str:
    return """
    Extract from the answer the House Speaker elected at the beginning of the 119th Congress (January 2025).
    Return a top-level key `speaker` with:
    - name: the Speaker’s name (string)
    - election_date: the election date as presented (string)
    - final_vote_count: the final vote count (string, e.g., "218-215")
    - vote_details: any textual characterization of the vote (string; e.g., "decided on first ballot despite initial holdouts")
    - sources: a list of all URLs cited in the answer for this Speaker (including House.gov, Congress.gov, or news pages). Extract only actual URLs present in the answer.

    If any field is missing, set it to null; if no URLs are cited, return an empty list for `sources`.
    Do not invent any information not explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


def _position_list_to_text(positions: List[str]) -> str:
    if not positions:
        return ""
    if len(positions) == 1:
        return positions[0]
    return ", ".join(positions[:-1]) + " or " + positions[-1]


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_cabinet_group(
    evaluator: Evaluator,
    parent_node,
    group_id: str,
    group_desc: str,
    official: Optional[CabinetOfficial],
    expected_positions: List[str],
    expected_date: str,
    expected_vote: str,
    expected_tiebreaker: str,
) -> None:
    """
    Build the verification subtree for a single cabinet official group (first/second/third).
    """
    grp_node = evaluator.add_parallel(
        id=f"{group_id}_confirmed_cabinet",
        desc=group_desc,
        parent=parent_node,
        critical=False,
    )

    # Existence of at least one source URL for this official (gate further checks)
    sources_exist = bool(official and official.sources and len(official.sources) > 0)
    evaluator.add_custom_node(
        result=sources_exist,
        id=f"{group_id}_sources_provided",
        desc=f"{group_desc}: At least one reference source URL is provided",
        parent=grp_node,
        critical=True,  # Gate all other checks
    )

    # Position title matches expected (e.g., Secretary of State / Defense / Attorney General)
    pos_leaf = evaluator.add_leaf(
        id=f"{group_id}_position_title",
        desc=f"Cabinet position title matches {_position_list_to_text(expected_positions)}",
        parent=grp_node,
        critical=True,
    )
    claim_pos = (
        f"The official discussed was confirmed to the role {_position_list_to_text(expected_positions)}."
    )
    await evaluator.verify(
        claim=claim_pos,
        node=pos_leaf,
        sources=_safe_sources(official.sources if official else []),
        additional_instruction=(
            "Confirm from the provided page(s) that the Cabinet role is exactly one of the expected titles. "
            "Allow minor synonyms or historical equivalents if applicable (e.g., 'Secretary of War' treated as equivalent in context to 'Secretary of Defense'). "
            "If the page does not clearly indicate the role, mark as not supported."
        ),
    )

    # Official name correctly provided
    name_leaf = evaluator.add_leaf(
        id=f"{group_id}_official_name",
        desc="Official's name is correctly provided",
        parent=grp_node,
        critical=True,
    )
    name_txt = official.name if official and official.name else ""
    claim_name = f"The person confirmed to {_position_list_to_text(expected_positions)} was {name_txt}."
    await evaluator.verify(
        claim=claim_name,
        node=name_leaf,
        sources=_safe_sources(official.sources if official else []),
        additional_instruction=(
            "Verify that the source page explicitly names the confirmed official for the specified role. "
            "Allow minor name variants (middle initials, casing). If the source does not affirm the name, mark as not supported."
        ),
    )

    # Confirmation date matches expected
    date_leaf = evaluator.add_leaf(
        id=f"{group_id}_confirmation_date",
        desc=f"Confirmation date matches {expected_date}",
        parent=grp_node,
        critical=True,
    )
    claim_date = (
        f"The Senate confirmation (final vote) for {name_txt} occurred on {expected_date}."
    )
    await evaluator.verify(
        claim=claim_date,
        node=date_leaf,
        sources=_safe_sources(official.sources if official else []),
        additional_instruction=(
            "Check the page to confirm the Senate confirmation date (the date of the vote) matches the expected date exactly. "
            "Accept reasonable date formatting variants but the calendar date must be the same."
        ),
    )

    # Final Senate vote count matches expected (e.g., 99-0, 51-50, 54-46)
    vote_leaf = evaluator.add_leaf(
        id=f"{group_id}_vote_count",
        desc=f"Senate vote count matches {expected_vote}",
        parent=grp_node,
        critical=True,
    )
    claim_vote = f"The final Senate vote count for this confirmation was {expected_vote}."
    await evaluator.verify(
        claim=claim_vote,
        node=vote_leaf,
        sources=_safe_sources(official.sources if official else []),
        additional_instruction=(
            "Verify the final roll call tally on the source page. "
            "Accept common formats (e.g., 'Yea–Nay', 'Yes–No'). The numbers must match the expected count."
        ),
    )

    # Vote characterization (non-critical): unanimous / narrow or tie
    if group_id == "first":
        vote_nature_leaf = evaluator.add_leaf(
            id=f"{group_id}_vote_nature",
            desc="Correctly indicates the vote was unanimous",
            parent=grp_node,
            critical=False,
        )
        claim_nature = (
            "The confirmation vote was unanimous (no 'nay' votes recorded)."
        )
    elif group_id == "second":
        vote_nature_leaf = evaluator.add_leaf(
            id=f"{group_id}_vote_margin",
            desc="Correctly identifies this as a narrow/tie vote requiring tiebreaker",
            parent=grp_node,
            critical=False,
        )
        claim_nature = (
            "The vote was tied or effectively a 50–50 situation that required a Vice Presidential tie‑breaking vote."
        )
    else:  # third
        vote_nature_leaf = evaluator.add_leaf(
            id=f"{group_id}_vote_margin",
            desc="Correctly characterizes the vote margin (e.g., partisan, narrow majority)",
            parent=grp_node,
            critical=False,
        )
        claim_nature = (
            "The final vote margin reflects a non‑unanimous, partisan or narrow majority outcome (e.g., 54–46)."
        )

    await evaluator.verify(
        claim=claim_nature,
        node=vote_nature_leaf,
        sources=_safe_sources(official.sources if official else []),
        additional_instruction=(
            "Determine from the source page whether the qualitative characterization of the vote (unanimity, tie, narrow/partisan) is accurate."
        ),
    )

    # Tiebreaker status (critical)
    tiebreak_leaf = evaluator.add_leaf(
        id=f"{group_id}_tiebreaker_status",
        desc=(
            "Correctly indicates Vice Presidential tie‑breaking vote was cast"
            if expected_tiebreaker.lower() == "yes"
            else "Correctly indicates no Vice Presidential tie‑breaking vote occurred"
        ),
        parent=grp_node,
        critical=True,
    )
    if expected_tiebreaker.lower() == "yes":
        claim_tie = (
            "A Vice Presidential tie‑breaking vote was cast for this confirmation."
        )
    else:
        claim_tie = (
            "No Vice Presidential tie‑breaking vote occurred for this confirmation."
        )

    await evaluator.verify(
        claim=claim_tie,
        node=tiebreak_leaf,
        sources=_safe_sources(official.sources if official else []),
        additional_instruction=(
            "Check whether the Vice President cast a tie‑breaking vote on the source page. "
            "If the page is silent or ambiguous, mark as not supported."
        ),
    )

    # Reference URL validity/support (critical): page supports the confirmation details
    ref_leaf = evaluator.add_leaf(
        id=f"{group_id}_reference_url",
        desc="Valid reference URL from Senate.gov, Congress.gov, or reputable news source provided",
        parent=grp_node,
        critical=True,
    )
    claim_ref = (
        f"The provided reference URL(s) report the Senate confirmation of {name_txt} to {_position_list_to_text(expected_positions)}."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=_safe_sources(official.sources if official else []),
        additional_instruction=(
            "Confirm that at least one provided URL meaningfully covers the confirmation (role, person, vote/date). "
            "Prefer Senate.gov/Congress.gov; reputable mainstream news (e.g., AP, Reuters, major national outlets) is acceptable."
        ),
    )


async def verify_speaker_group(
    evaluator: Evaluator,
    parent_node,
    speaker: Optional[SpeakerInfo],
    group_desc: str,
    expected_date: str,
    expected_vote: str,
) -> None:
    """
    Build the verification subtree for the House Speaker election group.
    """
    grp_node = evaluator.add_parallel(
        id="house_speaker_election",
        desc=group_desc,
        parent=parent_node,
        critical=False,
    )

    sources_exist = bool(speaker and speaker.sources and len(speaker.sources) > 0)
    evaluator.add_custom_node(
        result=sources_exist,
        id="speaker_sources_provided",
        desc="House Speaker group: At least one reference source URL is provided",
        parent=grp_node,
        critical=True,
    )

    # Speaker name
    name_leaf = evaluator.add_leaf(
        id="speaker_name",
        desc="Speaker's name is correctly provided",
        parent=grp_node,
        critical=True,
    )
    speaker_name = speaker.name if speaker and speaker.name else ""
    claim_name = (
        f"The House elected {speaker_name} as Speaker at the beginning of the 119th Congress."
    )
    await evaluator.verify(
        claim=claim_name,
        node=name_leaf,
        sources=_safe_sources(speaker.sources if speaker else []),
        additional_instruction=(
            "Verify from the provided page(s) that the person named was elected Speaker at the start of the 119th Congress."
        ),
    )

    # Election date
    date_leaf = evaluator.add_leaf(
        id="speaker_election_date",
        desc=f"Election date matches {expected_date}",
        parent=grp_node,
        critical=True,
    )
    claim_date = f"The Speaker election occurred on {expected_date}."
    await evaluator.verify(
        claim=claim_date,
        node=date_leaf,
        sources=_safe_sources(speaker.sources if speaker else []),
        additional_instruction=(
            "Check the page for the calendar date of the Speaker election. "
            "Accept formatting variants but the date must match exactly."
        ),
    )

    # Final vote count
    vote_leaf = evaluator.add_leaf(
        id="speaker_vote_count",
        desc=f"Final vote count matches {expected_vote}",
        parent=grp_node,
        critical=True,
    )
    claim_vote = f"The final vote count for the Speaker election was {expected_vote}."
    await evaluator.verify(
        claim=claim_vote,
        node=vote_leaf,
        sources=_safe_sources(speaker.sources if speaker else []),
        additional_instruction=(
            "Verify the final tally on the provided page(s)."
        ),
    )

    # Vote details (non-critical)
    details_leaf = evaluator.add_leaf(
        id="speaker_vote_details",
        desc="Correctly indicates this was decided on first ballot despite initial holdouts",
        parent=grp_node,
        critical=False,
    )
    details_txt = speaker.vote_details if speaker and speaker.vote_details else ""
    claim_details = (
        "The election was decided on the first ballot despite initial holdouts or intra‑party uncertainty."
    )
    await evaluator.verify(
        claim=claim_details,
        node=details_leaf,
        sources=_safe_sources(speaker.sources if speaker else []),
        additional_instruction=(
            "Assess the narrative description on the page(s). If they indicate a multi‑ballot process, mark as not supported."
        ),
    )

    # Reference URL validity/support (critical)
    ref_leaf = evaluator.add_leaf(
        id="speaker_reference_url",
        desc="Valid reference URL from House.gov, Congress.gov, or reputable news source provided",
        parent=grp_node,
        critical=True,
    )
    claim_ref = f"The provided reference URL(s) report the Speaker election of {speaker_name}."
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=_safe_sources(speaker.sources if speaker else []),
        additional_instruction=(
            "Confirm that at least one provided URL meaningfully covers the Speaker election (name, date, vote). "
            "Prefer House.gov/Congress.gov; reputable mainstream news is acceptable."
        ),
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
    Evaluate the answer for Trump administration's 2025 Cabinet confirmations and House Speaker election.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=ROOT_DESC,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Record ground-truth expectations (as rubric targets)
    evaluator.add_ground_truth(
        {
            "cabinet_expectations": CABINET_EXPECTATIONS,
            "speaker_expectations": SPEAKER_EXPECTATIONS,
        },
        gt_type="rubric_expectations",
    )

    # Extract structured information in two steps to keep results clear
    extraction_cabinet = await evaluator.extract(
        prompt=prompt_extract_cabinet(),
        template_class=FullExtraction,
        extraction_name="cabinet_triplet",
    )
    extraction_speaker = await evaluator.extract(
        prompt=prompt_extract_speaker(),
        template_class=FullExtraction,
        extraction_name="speaker_info",
    )

    # Merge extracted structures safely
    cabinet = None
    speaker = None

    # cabinet data may be in first extraction's `.cabinet`
    if extraction_cabinet and extraction_cabinet.cabinet:
        cabinet = extraction_cabinet.cabinet

    # speaker data may be in second extraction's `.speaker`
    if extraction_speaker and extraction_speaker.speaker:
        speaker = extraction_speaker.speaker

    # Build Cabinet groups under root
    # First
    first_official = cabinet.first if (cabinet and cabinet.first) else None
    await verify_cabinet_group(
        evaluator=evaluator,
        parent_node=root,
        group_id=CABINET_EXPECTATIONS[0]["group_id"],
        group_desc=CABINET_EXPECTATIONS[0]["group_desc"],
        official=first_official,
        expected_positions=CABINET_EXPECTATIONS[0]["expected_position"],
        expected_date=CABINET_EXPECTATIONS[0]["expected_date"],
        expected_vote=CABINET_EXPECTATIONS[0]["expected_vote"],
        expected_tiebreaker=CABINET_EXPECTATIONS[0]["expected_tiebreaker"],
    )

    # Second
    second_official = cabinet.second if (cabinet and cabinet.second) else None
    await verify_cabinet_group(
        evaluator=evaluator,
        parent_node=root,
        group_id=CABINET_EXPECTATIONS[1]["group_id"],
        group_desc=CABINET_EXPECTATIONS[1]["group_desc"],
        official=second_official,
        expected_positions=CABINET_EXPECTATIONS[1]["expected_position"],
        expected_date=CABINET_EXPECTATIONS[1]["expected_date"],
        expected_vote=CABINET_EXPECTATIONS[1]["expected_vote"],
        expected_tiebreaker=CABINET_EXPECTATIONS[1]["expected_tiebreaker"],
    )

    # Third
    third_official = cabinet.third if (cabinet and cabinet.third) else None
    await verify_cabinet_group(
        evaluator=evaluator,
        parent_node=root,
        group_id=CABINET_EXPECTATIONS[2]["group_id"],
        group_desc=CABINET_EXPECTATIONS[2]["group_desc"],
        official=third_official,
        expected_positions=CABINET_EXPECTATIONS[2]["expected_position"],
        expected_date=CABINET_EXPECTATIONS[2]["expected_date"],
        expected_vote=CABINET_EXPECTATIONS[2]["expected_vote"],
        expected_tiebreaker=CABINET_EXPECTATIONS[2]["expected_tiebreaker"],
    )

    # Speaker group
    await verify_speaker_group(
        evaluator=evaluator,
        parent_node=root,
        speaker=speaker,
        group_desc=SPEAKER_EXPECTATIONS["group_desc"],
        expected_date=SPEAKER_EXPECTATIONS["expected_date"],
        expected_vote=SPEAKER_EXPECTATIONS["expected_vote"],
    )

    return evaluator.get_summary()