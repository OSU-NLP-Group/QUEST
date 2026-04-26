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
TASK_ID = "ewc2025_cs2_darkhan_player"
TASK_DESCRIPTION = (
    "In the Counter-Strike 2 championship held at the Esports World Cup 2025 in Riyadh, Saudi Arabia in August 2025, "
    "identify the winning team. From that team's roster, find the player who was born in Darkhan, Mongolia. Provide this "
    "player's complete birth date (including year, month, and day) and include a reference URL that verifies this information."
)

# Ground truth expectations (used for answer-content checks and guidance)
EXPECTED = {
    "event_identity": "Esports World Cup 2025 Counter-Strike 2 championship in Riyadh, Saudi Arabia",
    "event_dates": "August 20–24, 2025",
    "champion": "The MongolZ",
    "final_score": "3–0",
    "player": "Usukhbayar '910' Banzragch",
    "birthplace": "Darkhan, Mongolia",
    "birthdate": "July 5, 2002",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    name_or_label: Optional[str] = None
    location: Optional[str] = None
    dates: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ChampionInfo(BaseModel):
    team: Optional[str] = None
    grand_final_score: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PlayerInfo(BaseModel):
    full_name: Optional[str] = None
    # URLs specifically used to support roster membership for this event
    roster_sources: List[str] = Field(default_factory=list)
    birthplace: Optional[str] = None
    birthplace_sources: List[str] = Field(default_factory=list)
    birthdate: Optional[str] = None
    birthdate_sources: List[str] = Field(default_factory=list)


class CS2EWC2025Extraction(BaseModel):
    event: Optional[EventInfo] = None
    champion: Optional[ChampionInfo] = None
    player: Optional[PlayerInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cs2_ewc2025() -> str:
    return """
    Extract the key facts and the exact URL citations that the answer provides.

    Fields to extract (return null for missing scalar fields and [] for missing lists):
    - event:
        - name_or_label: The tournament name/label as stated in the answer (e.g., "Esports World Cup 2025 Counter-Strike 2")
        - location: The event location as stated (e.g., "Riyadh, Saudi Arabia")
        - dates: The event dates as stated (e.g., "August 20–24, 2025" or "Aug 20-24, 2025")
        - sources: A list of URLs the answer uses to support the event identity and/or dates
    - champion:
        - team: The champion team as stated (e.g., "The MongolZ")
        - grand_final_score: The grand-final score as stated (e.g., "3–0" or "3-0")
        - sources: A list of URLs cited to support the champion and/or the final score
    - player:
        - full_name: The selected player's full name as stated (e.g., "Usukhbayar '910' Banzragch")
        - roster_sources: URLs cited to support that this player was on The MongolZ roster for the event
        - birthplace: The player's birthplace as stated (e.g., "Darkhan, Mongolia")
        - birthplace_sources: URLs cited to support the birthplace
        - birthdate: The player's birthdate as stated (e.g., "July 5, 2002" or "2002-07-05")
        - birthdate_sources: URLs cited to support the birthdate

    Special URL extraction rules:
    - Extract only explicit URLs present in the answer (including markdown links).
    - Normalize to full URLs including protocol.
    - If a category has no URLs provided, return an empty list for that category.
    """


# --------------------------------------------------------------------------- #
# Helper for URL-supported verification groups                                #
# --------------------------------------------------------------------------- #
async def add_url_support_group(
    evaluator: Evaluator,
    parent,
    *,
    group_id: str,
    group_desc: str,
    urls_exist_id: str,
    urls_exist_desc: str,
    verify_id: str,
    verify_desc: str,
    claim: str,
    urls: List[str],
    additional_instruction: str,
):
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent,
        critical=True,
    )

    # Existence check (critical) to ensure URL grounding
    urls_exist = evaluator.add_custom_node(
        result=bool(urls),
        id=urls_exist_id,
        desc=urls_exist_desc,
        parent=group_node,
        critical=True,
    )

    # Actual URL-based verification (will auto-skip if urls_exist fails due to critical sibling)
    verify_node = evaluator.add_leaf(
        id=verify_id,
        desc=verify_desc,
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=urls,  # multi-URL verification; passes if any one URL supports the claim
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Tree construction and verifications                                         #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, main_node, extracted: CS2EWC2025Extraction) -> None:
    # Normalize extracted blocks
    event = extracted.event or EventInfo()
    champion = extracted.champion or ChampionInfo()
    player = extracted.player or PlayerInfo()

    # 1) EventIdentification (parallel, critical)
    event_node = evaluator.add_parallel(
        id="EventIdentification",
        desc="Correctly identifies the tournament as the Counter-Strike 2 championship at Esports World Cup 2025 in Riyadh, Saudi Arabia, occurring Aug 20–24, 2025.",
        parent=main_node,
        critical=True,
    )

    # 1.a) Event identity/location is correctly stated in the answer (simple check against the answer text)
    event_identity_leaf = evaluator.add_leaf(
        id="EventIsEWC2025CS2InRiyadh",
        desc="States the event is the Esports World Cup 2025 Counter-Strike 2 championship in Riyadh, Saudi Arabia.",
        parent=event_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly identifies the tournament as the Esports World Cup 2025 Counter-Strike 2 championship held in Riyadh, Saudi Arabia.",
        node=event_identity_leaf,
        additional_instruction="Check the assistant's answer text only. Reasonable wording variations are allowed; the meaning must be clear.",
    )

    # 1.b) Event dates are correctly stated in the answer (simple check)
    event_dates_leaf = evaluator.add_leaf(
        id="EventDatesAreAug20ToAug24_2025",
        desc="States the event dates are August 20–24, 2025.",
        parent=event_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the event took place on August 20–24, 2025 (accept minor formatting variants like 'Aug 20-24, 2025').",
        node=event_dates_leaf,
        additional_instruction="Focus on the assistant's answer text; allow 'Aug' vs 'August' and hyphen/en dash variations.",
    )

    # 2) ChampionAndFinalResult (parallel, critical)
    champion_node = evaluator.add_parallel(
        id="ChampionAndFinalResult",
        desc="Correctly states the championship-winning team and the grand-final score per constraints.",
        parent=main_node,
        critical=True,
    )

    # 2.a) Champion team is The MongolZ (simple check against answer)
    champion_leaf = evaluator.add_leaf(
        id="WinningTeamIsTheMongolZ",
        desc="Names The MongolZ as the championship-winning team.",
        parent=champion_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer names 'The MongolZ' as the champion of the Esports World Cup 2025 CS2 event.",
        node=champion_leaf,
        additional_instruction="Check the assistant's answer text for the champion team identification.",
    )

    # 2.b) Grand final score is 3–0 (simple check against answer)
    final_score_leaf = evaluator.add_leaf(
        id="GrandFinalScoreIs3to0",
        desc="States The MongolZ won the grand finals by a 3–0 score.",
        parent=champion_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that The MongolZ won the grand final by a 3–0 score (accept '3-0' vs '3–0').",
        node=final_score_leaf,
        additional_instruction="Look only at the assistant's answer text; allow hyphen/en dash variations and minor spacing differences.",
    )

    # 3) DarkhanBornPlayerSelection (parallel, critical)
    player_node = evaluator.add_parallel(
        id="DarkhanBornPlayerSelection",
        desc="Selects the constrained player from the champion roster and states the required birthplace detail.",
        parent=main_node,
        critical=True,
    )

    # 3.a) Selected player identity (simple check)
    player_identity_leaf = evaluator.add_leaf(
        id="SelectedPlayerIsUsukhbayar910Banzragch",
        desc="Identifies the player as Usukhbayar '910' Banzragch.",
        parent=player_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer identifies the player as Usukhbayar '910' Banzragch (allow minor formatting around the nickname).",
        node=player_identity_leaf,
        additional_instruction="Focus on the assistant's answer text. Allow small variations like without quotes or different apostrophes.",
    )

    # 3.b) Player is on The MongolZ roster for the event (simple check)
    roster_membership_leaf = evaluator.add_leaf(
        id="PlayerIsOnTheMongolZRosterForEvent",
        desc="Indicates the selected player is on The MongolZ roster for the event.",
        parent=player_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer indicates that Usukhbayar '910' Banzragch was on The MongolZ roster for the Esports World Cup 2025 CS2 event.",
        node=roster_membership_leaf,
        additional_instruction="Focus on the assistant's answer text. Accept clear statements of roster membership for this event.",
    )

    # 3.c) Player birthplace is Darkhan, Mongolia (simple check)
    birthplace_leaf = evaluator.add_leaf(
        id="PlayerBirthplaceIsDarkhanMongolia",
        desc="States the selected player's birthplace is Darkhan, Mongolia.",
        parent=player_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the player's birthplace is Darkhan, Mongolia (accept 'Darkhan-Uul' variants referring to the same place).",
        node=birthplace_leaf,
        additional_instruction="Check the assistant's answer text; allow 'Darkhan-Uul' as equivalent to Darkhan for birthplace.",
    )

    # 4) BirthDate (parallel, critical)
    birthdate_node = evaluator.add_parallel(
        id="BirthDate",
        desc="Provides the player's complete birth date exactly as constrained.",
        parent=main_node,
        critical=True,
    )

    birthdate_leaf = evaluator.add_leaf(
        id="BirthDateIsJuly5_2002",
        desc="States the player's birth date is July 5, 2002 (year, month, and day included unambiguously).",
        parent=birthdate_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer provides the player's full birth date as July 5, 2002 (allow variants like 5 July 2002 or 2002-07-05).",
        node=birthdate_leaf,
        additional_instruction="Focus on the assistant's answer text. The day, month, and year must be unambiguously provided.",
    )

    # 5) SourceURLs (parallel, critical) — URL-grounded checks for each factual area
    sources_node = evaluator.add_parallel(
        id="SourceURLs",
        desc="Provides verifiable URL evidence for each required factual claim (event details, champion + 3–0, player roster, birthplace, and birth date).",
        parent=main_node,
        critical=True,
    )

    # 5.a) Event identity and dates supported by URLs
    await add_url_support_group(
        evaluator,
        sources_node,
        group_id="EventURLsGroup",
        group_desc="Event URLs group (identity + dates).",
        urls_exist_id="EventURLsProvided",
        urls_exist_desc="At least one event URL is provided.",
        verify_id="URLSupportsEventIdentityAndDates",
        verify_desc="Includes at least one URL that supports the event identity (EWC 2025 CS2 in Riyadh) and the Aug 20–24, 2025 dates.",
        claim="This page confirms the Esports World Cup 2025 Counter-Strike 2 event in Riyadh, Saudi Arabia and shows the event dates as August 20–24, 2025 (local time).",
        urls=event.sources,
        additional_instruction="Confirm both the tournament identity (EWC 2025 CS2 in Riyadh) and that the scheduled dates are August 20–24, 2025. Allow minor date formatting variants like 'Aug 20-24, 2025'.",
    )

    # 5.b) Champion and 3–0 result supported by URLs
    await add_url_support_group(
        evaluator,
        sources_node,
        group_id="ChampionURLsGroup",
        group_desc="Champion/result URLs group.",
        urls_exist_id="ChampionURLsProvided",
        urls_exist_desc="At least one champion/result URL is provided.",
        verify_id="URLSupportsChampionAnd3to0Result",
        verify_desc="Includes at least one URL that supports The MongolZ as champion and the 3–0 grand-final result.",
        claim="This page confirms that The MongolZ won the Esports World Cup 2025 CS2 championship and that the grand-final score was 3–0.",
        urls=champion.sources,
        additional_instruction="The page should explicitly or clearly indicate the champion (The MongolZ) and the 3–0 final score; allow '3-0' vs '3–0'.",
    )

    # 5.c) Player roster membership supported by URLs
    await add_url_support_group(
        evaluator,
        sources_node,
        group_id="RosterURLsGroup",
        group_desc="Player roster membership URLs group.",
        urls_exist_id="RosterURLsProvided",
        urls_exist_desc="At least one roster URL is provided for the player.",
        verify_id="URLSupportsPlayerRoster",
        verify_desc="Includes at least one URL that supports that Usukhbayar '910' Banzragch is on The MongolZ roster (preferably for the event).",
        claim="This page confirms that Usukhbayar '910' Banzragch was on The MongolZ roster for the Esports World Cup 2025 CS2 event (or clearly a The MongolZ player at that time).",
        urls=player.roster_sources,
        additional_instruction="Prefer event-specific roster pages. If not available, accept authoritative pages showing he was a The MongolZ player during that period.",
    )

    # 5.d) Player birthplace supported by URLs
    await add_url_support_group(
        evaluator,
        sources_node,
        group_id="BirthplaceURLsGroup",
        group_desc="Player birthplace URLs group.",
        urls_exist_id="BirthplaceURLsProvided",
        urls_exist_desc="At least one birthplace URL is provided for the player.",
        verify_id="URLSupportsPlayerBirthplace",
        verify_desc="Includes at least one URL that supports that his birthplace is Darkhan, Mongolia.",
        claim="This page confirms that Usukhbayar '910' Banzragch's birthplace is Darkhan, Mongolia.",
        urls=player.birthplace_sources,
        additional_instruction="Allow variants like 'Darkhan-Uul' referring to Darkhan, Mongolia.",
    )

    # 5.e) Player birthdate supported by URLs
    await add_url_support_group(
        evaluator,
        sources_node,
        group_id="BirthdateURLsGroup",
        group_desc="Player birthdate URLs group.",
        urls_exist_id="BirthdateURLsProvided",
        urls_exist_desc="At least one birthdate URL is provided for the player.",
        verify_id="URLSupportsBirthDate",
        verify_desc="Includes at least one URL that explicitly supports the stated birth date (July 5, 2002).",
        claim="This page confirms that Usukhbayar '910' Banzragch was born on July 5, 2002 (accept 5 July 2002 or 2002-07-05 as equivalent).",
        urls=player.birthdate_sources,
        additional_instruction="The date must clearly include year, month, and day. Accept reasonable format variations.",
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
    Evaluate an answer for the Esports World Cup 2025 CS2 winner and Darkhan-born player birthdate task.
    """
    # Initialize evaluator (root is non-critical by design; we add a critical main node under it)
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

    # Add main critical sequential node to mirror the rubric's critical root
    main = evaluator.add_sequential(
        id="Root",
        desc="Answer satisfies all constraints: identifies the specified CS2 event at Esports World Cup 2025 (Riyadh, Aug 20–24, 2025), states the champion and grand-final result, identifies the specified Darkhan-born player from the champion roster, provides the player's complete birth date, and supplies verifiable URLs supporting each factual claim.",
        parent=root,
        critical=True,
    )

    # Extract structured info and URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_cs2_ewc2025(),
        template_class=CS2EWC2025Extraction,
        extraction_name="cs2_ewc2025_extraction",
    )

    # Record ground truth expectations for transparency
    evaluator.add_ground_truth(
        {
            "expected_event_identity": EXPECTED["event_identity"],
            "expected_event_dates": EXPECTED["event_dates"],
            "expected_champion": EXPECTED["champion"],
            "expected_final_score": EXPECTED["final_score"],
            "expected_player": EXPECTED["player"],
            "expected_birthplace": EXPECTED["birthplace"],
            "expected_birthdate": EXPECTED["birthdate"],
        },
        gt_type="expected_constraints",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, main, extracted)

    # Return final summary
    return evaluator.get_summary()