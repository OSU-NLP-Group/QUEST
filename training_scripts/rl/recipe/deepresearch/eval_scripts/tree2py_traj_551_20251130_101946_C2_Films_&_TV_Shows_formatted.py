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
TASK_ID = "dwts_winner_2025_criteria"
TASK_DESCRIPTION = (
    "Identify the winner of a Dancing with the Stars season whose finale aired in 2025, "
    "who meets the following criteria: born in December 2003, currently works as a wildlife "
    "conservationist at a zoo in Queensland, Australia, and is the child of a famous wildlife "
    "expert who died in 2006. Provide the winner's name, their professional dance partner's name, "
    "the season number they won, and the exact date the finale aired on ABC and Disney+."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DWTSAnswerExtraction(BaseModel):
    # Required outputs
    winner_name: Optional[str] = None
    partner_name: Optional[str] = None
    season_number: Optional[str] = None
    finale_date: Optional[str] = None  # e.g., "March 10, 2025" (string format is fine)
    broadcast_networks_text: Optional[str] = None  # e.g., "ABC and Disney+"

    # Background constraints
    birth_date_text: Optional[str] = None  # e.g., "December 1, 2003" or "December 2003"
    family_background_text: Optional[str] = None  # e.g., "son of Steve Irwin, who died in 2006"
    professional_role_text: Optional[str] = None  # e.g., "wildlife conservationist at Australia Zoo in Queensland"

    # URLs explicitly mentioned in the answer
    primary_sources: List[str] = Field(default_factory=list)

    # Optional per-claim sources (subset of primary_sources or other cited links)
    winner_sources: List[str] = Field(default_factory=list)
    season_sources: List[str] = Field(default_factory=list)
    partner_sources: List[str] = Field(default_factory=list)
    finale_sources: List[str] = Field(default_factory=list)
    broadcast_sources: List[str] = Field(default_factory=list)
    birth_sources: List[str] = Field(default_factory=list)
    family_sources: List[str] = Field(default_factory=list)
    role_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_dwts_info() -> str:
    return """
    Extract the requested information from the provided answer text exactly as stated. Return null for any field not present.
    Required fields:
    - winner_name: The name of the Dancing with the Stars (U.S.) winner referenced in the answer.
    - partner_name: The professional dance partner's name associated with the same win.
    - season_number: The season number won by the winner (string; do not coerce to integer).
    - finale_date: The exact date the finale aired (string as given; e.g., 'March 10, 2025').
    - broadcast_networks_text: The text describing where the finale aired (e.g., 'ABC and Disney+').
    Background verification text:
    - birth_date_text: The birth date text for the winner (e.g., 'December 1, 2003' or 'December 2003').
    - family_background_text: A short phrase describing the parent relationship and death year (e.g., 'child of Steve Irwin who died in 2006').
    - professional_role_text: A short phrase describing the winner's role in wildlife conservation at a zoo in Queensland (e.g., 'wildlife conservationist at Australia Zoo in Queensland').
    URL sources:
    - primary_sources: List all URLs explicitly cited anywhere in the answer.
    - winner_sources: URLs cited that directly support the winner identity claim (subset of primary_sources if applicable).
    - season_sources: URLs cited that directly support the season number and its finale timing (subset of primary_sources if applicable).
    - partner_sources: URLs cited that directly support the professional partner claim.
    - finale_sources: URLs cited that directly support the finale date claim.
    - broadcast_sources: URLs cited that directly support the 'aired on ABC and Disney+' claim.
    - birth_sources: URLs cited that directly support the birth date claim.
    - family_sources: URLs cited that directly support the family background (child of a wildlife expert who died in 2006) claim.
    - role_sources: URLs cited that directly support the 'wildlife conservationist at a zoo in Queensland' claim.
    Notes:
    - Only include URLs that are explicitly present in the answer (plain links or markdown-style links).
    - Do not fabricate or infer new URLs.
    - If a given per-claim source list is not present in the answer, return an empty list for that field.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _coalesce_sources(*lists: Optional[List[str]]) -> List[str]:
    """Merge multiple source lists into a unique, ordered list (preserve first-seen order)."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if isinstance(url, str) and url.strip() and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def _safe(val: Optional[str]) -> str:
    return val or ""


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_dwts_tree(evaluator: Evaluator, root_node, data: DWTSAnswerExtraction) -> None:
    """
    Build the rubric tree and verify each leaf according to the given specification.
    All children under the critical root are also critical. Each leaf is a single verification step.
    """
    # Critical parent: DWTS_Winner_Verification
    top = evaluator.add_parallel(
        id="DWTS_Winner_Verification",
        desc="Verify all required information about the Dancing with the Stars winner who meets the specified criteria",
        parent=root_node,
        critical=True
    )

    # Sub-node: Winner_and_Season (critical, parallel)
    winner_and_season = evaluator.add_parallel(
        id="Winner_and_Season",
        desc="Verify the winner's identity and the season they won",
        parent=top,
        critical=True
    )

    # Leaves under Winner_and_Season
    node_winner_name = evaluator.add_leaf(
        id="Winner_Name",
        desc="Provide the correct name of the DWTS winner who meets all specified criteria",
        parent=winner_and_season,
        critical=True
    )
    node_season_number = evaluator.add_leaf(
        id="Season_Number",
        desc="Identify the correct DWTS season number with finale in 2025",
        parent=winner_and_season,
        critical=True
    )

    # Sub-node: Competition_Details (critical, parallel)
    competition_details = evaluator.add_parallel(
        id="Competition_Details",
        desc="Verify competition broadcast and partnership details",
        parent=top,
        critical=True
    )

    # Leaves under Competition_Details
    node_partner_name = evaluator.add_leaf(
        id="Partner_Name",
        desc="Provide the professional dance partner's name",
        parent=competition_details,
        critical=True
    )
    node_finale_date = evaluator.add_leaf(
        id="Finale_Date",
        desc="Provide the exact date the finale aired",
        parent=competition_details,
        critical=True
    )
    node_broadcast_network = evaluator.add_leaf(
        id="Broadcast_Network",
        desc="Confirm the finale aired on ABC and Disney+",
        parent=competition_details,
        critical=True
    )

    # Sub-node: Background_Verification (critical, parallel)
    background = evaluator.add_parallel(
        id="Background_Verification",
        desc="Verify the contestant's personal and professional background meets all specified criteria",
        parent=top,
        critical=True
    )

    # Leaves under Background_Verification
    node_birth_date = evaluator.add_leaf(
        id="Birth_Date",
        desc="Confirm the individual was born in December 2003",
        parent=background,
        critical=True
    )
    node_family_background = evaluator.add_leaf(
        id="Family_Background",
        desc="Confirm the individual is the child of a wildlife expert who died in 2006",
        parent=background,
        critical=True
    )
    node_professional_role = evaluator.add_leaf(
        id="Professional_Role",
        desc="Verify the individual works as a wildlife conservationist at a zoo in Queensland, Australia",
        parent=background,
        critical=True
    )

    # Build claims and corresponding sources
    winner_name = _safe(data.winner_name)
    partner_name = _safe(data.partner_name)
    season_number = _safe(data.season_number)
    finale_date = _safe(data.finale_date)

    # Winner name claim
    if season_number and finale_date:
        claim_winner = (
            f"On Dancing with the Stars (U.S.) season {season_number}, whose finale aired in 2025, "
            f"the winner was {winner_name}."
        )
    elif season_number:
        claim_winner = (
            f"On Dancing with the Stars (U.S.) season {season_number} (finale in 2025), the winner was {winner_name}."
        )
    elif finale_date:
        claim_winner = (
            f"The season of Dancing with the Stars (U.S.) whose finale aired on {finale_date} in 2025 "
            f"was won by {winner_name}."
        )
    else:
        claim_winner = (
            f"{winner_name} won a season of Dancing with the Stars (U.S.) whose finale aired in 2025."
        )

    winner_sources = _coalesce_sources(
        data.winner_sources, data.season_sources, data.finale_sources, data.primary_sources
    )

    # Season number claim
    if finale_date:
        claim_season = (
            f"On Dancing with the Stars (U.S.), the season whose finale aired on {finale_date} (in 2025) "
            f"is season {season_number}."
        )
    else:
        claim_season = (
            f"The Dancing with the Stars (U.S.) season that had its finale in 2025 is season {season_number}."
        )
    season_sources = _coalesce_sources(
        data.season_sources, data.finale_sources, data.primary_sources
    )

    # Partner name claim
    if season_number:
        claim_partner = (
            f"On Dancing with the Stars (U.S.) season {season_number}, the professional dance partner of "
            f"{winner_name} was {partner_name}."
        )
    else:
        claim_partner = (
            f"The professional dance partner of {winner_name} on Dancing with the Stars (U.S.) was {partner_name}."
        )
    partner_sources = _coalesce_sources(
        data.partner_sources, data.winner_sources, data.season_sources, data.primary_sources
    )

    # Finale date claim
    if season_number:
        claim_finale_date = (
            f"The finale of Dancing with the Stars (U.S.) season {season_number} aired on {finale_date}."
        )
    else:
        claim_finale_date = f"The Dancing with the Stars (U.S.) finale aired on {finale_date}."
    finale_sources = _coalesce_sources(
        data.finale_sources, data.season_sources, data.primary_sources
    )

    # Broadcast network claim
    if season_number and finale_date:
        claim_broadcast = (
            f"The finale of Dancing with the Stars (U.S.) season {season_number} on {finale_date} aired on ABC and Disney+."
        )
    elif season_number:
        claim_broadcast = (
            f"The finale of Dancing with the Stars (U.S.) season {season_number} aired on ABC and Disney+."
        )
    elif finale_date:
        claim_broadcast = (
            f"The finale of Dancing with the Stars (U.S.) on {finale_date} aired on ABC and Disney+."
        )
    else:
        claim_broadcast = "The finale of Dancing with the Stars (U.S.) aired on ABC and Disney+."
    broadcast_sources = _coalesce_sources(
        data.broadcast_sources, data.finale_sources, data.season_sources, data.primary_sources
    )

    # Birth date claim (must be December 2003)
    claim_birth = (
        f"{winner_name} was born in December 2003."
    )
    birth_sources = _coalesce_sources(
        data.birth_sources, data.primary_sources
    )

    # Family background claim (child of a wildlife expert who died in 2006)
    claim_family = (
        f"{winner_name} is the child of a famous wildlife expert who died in 2006."
    )
    family_sources = _coalesce_sources(
        data.family_sources, data.primary_sources
    )

    # Professional role claim (wildlife conservationist at a zoo in Queensland, Australia)
    claim_role = (
        f"{winner_name} currently works as a wildlife conservationist at a zoo in Queensland, Australia."
    )
    role_sources = _coalesce_sources(
        data.role_sources, data.primary_sources
    )

    # Prepare batch verifications
    claims_and_sources = [
        (
            claim_winner,
            winner_sources if winner_sources else None,
            node_winner_name,
            "Verify the named person is indeed the winner of the specified DWTS (U.S.) season that concluded in 2025. "
            "Allow minor formatting variants of names (e.g., middle initials)."
        ),
        (
            claim_season,
            season_sources if season_sources else None,
            node_season_number,
            "Verify that the DWTS (U.S.) season associated with a 2025 finale is indeed the specified season number."
        ),
        (
            claim_partner,
            partner_sources if partner_sources else None,
            node_partner_name,
            "Verify the professional partner as listed for the winning couple for that season."
        ),
        (
            claim_finale_date,
            finale_sources if finale_sources else None,
            node_finale_date,
            "Verify the exact U.S. broadcast date of the finale. Accept standard date formats; ensure the year is 2025."
        ),
        (
            claim_broadcast,
            broadcast_sources if broadcast_sources else None,
            node_broadcast_network,
            "Verify that the finale aired on ABC and was available on Disney+. Accept phrasing like 'aired on ABC and streamed on Disney+'."
        ),
        (
            claim_birth,
            birth_sources if birth_sources else None,
            node_birth_date,
            "Only verify the month and year (December 2003). The exact day may vary; any date in December 2003 counts as correct."
        ),
        (
            claim_family,
            family_sources if family_sources else None,
            node_family_background,
            "Verify that the winner is the child of a famous wildlife expert who died in 2006. "
            "If the page confirms the parent is Steve Irwin (d. 2006), that satisfies the criterion."
        ),
        (
            claim_role,
            role_sources if role_sources else None,
            node_professional_role,
            "Verify current or ongoing work as a wildlife conservationist at a zoo located in Queensland, Australia. "
            "Australia Zoo in Beerwah, Queensland qualifies."
        ),
    ]

    # Execute all verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the DWTS winner task with 2025-finale constraints.
    Returns an evaluation summary with the verification tree and scores.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall top-level aggregation is parallel per rubric
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_dwts_info(),
        template_class=DWTSAnswerExtraction,
        extraction_name="dwts_extraction",
    )

    # Add custom info for transparency (optional)
    evaluator.add_custom_info(
        {
            "criteria": {
                "finale_year": 2025,
                "birth_month_year_required": "December 2003",
                "parent_death_year_required": 2006,
                "role_required": "Wildlife conservationist at a zoo in Queensland, Australia",
                "broadcast_required": "Aired on ABC and Disney+",
            }
        },
        info_type="criteria",
        info_name="task_criteria"
    )

    # Build tree and verify all leaves
    await build_and_verify_dwts_tree(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()