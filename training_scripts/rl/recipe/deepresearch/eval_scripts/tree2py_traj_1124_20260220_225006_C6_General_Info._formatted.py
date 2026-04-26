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
TASK_ID = "wwe_nxt_smackdown_identification"
TASK_DESCRIPTION = (
    "Identify the WWE wrestler who meets all of the following criteria: "
    "(1) Won their first NXT Championship on April 23, 2024, by defeating Ilja Dragunov; "
    "(2) Competed in an NXT Championship match at NXT Vengeance Day 2024 on February 10, 2024; "
    "(3) Has held the NXT Championship twice during their career; "
    "(4) Was called up to the WWE main roster in December 2025 and assigned to SmackDown; "
    "(5) Made their SmackDown debut on January 9, 2026, in a match against Rey Fenix; "
    "(6) Won their SmackDown debut match; "
    "(7) Also held the TNA World Championship in 2025. "
    "Provide the wrestler's name along with supporting reference URLs that verify each of these achievements."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FirstNXTWinInfo(BaseModel):
    date: Optional[str] = None
    opponent: Optional[str] = None
    win_urls: List[str] = Field(default_factory=list)
    opponent_urls: List[str] = Field(default_factory=list)


class VengeanceDayInfo(BaseModel):
    event_date: Optional[str] = None
    match_type: Optional[str] = None
    event_urls: List[str] = Field(default_factory=list)


class NXTReignsInfo(BaseModel):
    total_reigns: Optional[str] = None  # Prefer string for flexibility ("2", "two")
    reigns_urls: List[str] = Field(default_factory=list)


class CallUpInfo(BaseModel):
    month_year: Optional[str] = None  # e.g., "December 2025"
    brand_assignment: Optional[str] = None  # e.g., "SmackDown"
    call_up_urls: List[str] = Field(default_factory=list)


class DebutInfo(BaseModel):
    debut_date: Optional[str] = None
    opponent: Optional[str] = None
    result: Optional[str] = None  # e.g., "won"
    debut_date_urls: List[str] = Field(default_factory=list)
    debut_opponent_urls: List[str] = Field(default_factory=list)
    debut_result_urls: List[str] = Field(default_factory=list)
    debut_general_urls: List[str] = Field(default_factory=list)  # fallback group if answer only gives one URL


class TNAInfo(BaseModel):
    tna_championship_year: Optional[str] = None  # e.g., "2025"
    tna_urls: List[str] = Field(default_factory=list)


class PreviousChampionContextInfo(BaseModel):
    dragunov_win_date: Optional[str] = None  # e.g., "September 30, 2023"
    dragunov_win_urls: List[str] = Field(default_factory=list)
    dragunov_reign_end_date: Optional[str] = None  # e.g., "April 23, 2024"
    reign_end_urls: List[str] = Field(default_factory=list)


class WrestlerExtraction(BaseModel):
    wrestler_name: Optional[str] = None

    first_nxt_win: FirstNXTWinInfo = Field(default_factory=FirstNXTWinInfo)
    vengeance_day: VengeanceDayInfo = Field(default_factory=VengeanceDayInfo)
    nxt_reigns: NXTReignsInfo = Field(default_factory=NXTReignsInfo)
    call_up: CallUpInfo = Field(default_factory=CallUpInfo)
    smackdown_debut: DebutInfo = Field(default_factory=DebutInfo)
    tna_info: TNAInfo = Field(default_factory=TNAInfo)
    previous_context: PreviousChampionContextInfo = Field(default_factory=PreviousChampionContextInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_wrestler() -> str:
    return """
    Extract the wrestler's name and structured facts exactly as stated in the provided answer text. Do NOT infer or add facts not present in the answer. Also extract the URLs the answer cites for each fact group.

    Return a JSON object with the following fields:

    - wrestler_name: the WWE wrestler's full name.

    - first_nxt_win: {
        date: date of the wrestler's FIRST NXT Championship win,
        opponent: opponent defeated to win that first NXT Championship,
        win_urls: array of URLs cited specifically for the first NXT title win (can include event recap pages, WWE/NXT pages, news articles, etc.),
        opponent_urls: array of URLs cited specifically for the opponent information (can overlap with win_urls if the answer does not separate them).
      }

    - vengeance_day: {
        event_date: date of NXT Vengeance Day 2024 (as cited in the answer),
        match_type: the description of the match type (e.g., "NXT Championship match") as stated,
        event_urls: array of URLs cited for the Vengeance Day participation.
      }

    - nxt_reigns: {
        total_reigns: the count or phrase indicating total NXT Championship reigns (e.g., "two"),
        reigns_urls: array of URLs cited for the total number of NXT Championship reigns.
      }

    - call_up: {
        month_year: the month and year of the WWE main roster call-up (e.g., "December 2025"),
        brand_assignment: the brand assigned upon call-up (e.g., "SmackDown"),
        call_up_urls: array of URLs cited for the call-up details.
      }

    - smackdown_debut: {
        debut_date: date of the SmackDown debut (e.g., "January 9, 2026"),
        opponent: opponent in the debut match (e.g., "Rey Fenix"),
        result: the result description (e.g., "won"),
        debut_date_urls: array of URLs cited specifically for debut date,
        debut_opponent_urls: array of URLs cited specifically for opponent info,
        debut_result_urls: array of URLs cited specifically for match result,
        debut_general_urls: array of general URLs cited for the debut if the answer provides a single page.
      }

    - tna_info: {
        tna_championship_year: the year the wrestler held the TNA World Championship (e.g., "2025"),
        tna_urls: array of URLs cited for the TNA championship claim.
      }

    - previous_context: {
        dragunov_win_date: date Ilja Dragunov won the NXT Championship prior to the wrestler's first win (e.g., "September 30, 2023"),
        dragunov_win_urls: array of URLs cited for Dragunov’s win,
        dragunov_reign_end_date: date Dragunov lost the NXT Championship (e.g., "April 23, 2024"),
        reign_end_urls: array of URLs cited for the end of Dragunov’s reign.
      }

    Rules:
    - Only extract URLs explicitly present in the answer text (including Markdown links). If the answer mentions a site (e.g., "Wikipedia") without a URL, return an empty array for that field.
    - If a field is not mentioned, set it to null (or empty array for URLs).
    - Do NOT invent or normalize dates—use the exact phrasing from the answer, even if it differs from standard formats.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _subject_name(extracted: WrestlerExtraction) -> str:
    return extracted.wrestler_name.strip() if extracted.wrestler_name else "the identified wrestler"


def _combine_urls(*url_lists: List[str]) -> List[str]:
    uniq = []
    seen = set()
    for lst in url_lists:
        for u in lst:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    uniq.append(uu)
                    seen.add(uu)
    return uniq


def _has_urls(urls: List[str]) -> bool:
    return bool([u for u in urls if isinstance(u, str) and u.strip()])


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_first_nxt_win(evaluator: Evaluator, parent, ex: WrestlerExtraction):
    node = evaluator.add_parallel(
        id="First_NXT_Title_Win",
        desc="Details of the wrestler's first NXT Championship victory",
        parent=parent,
        critical=True,
    )

    # URL existence for win date
    win_date_url_node = evaluator.add_custom_node(
        result=_has_urls(ex.first_nxt_win.win_urls),
        id="Win_Date_URL",
        desc="URL reference provided for the championship win date",
        parent=node,
        critical=True,
    )

    # Verify the win date claim
    win_date_verify_node = evaluator.add_leaf(
        id="Win_Date_Verification",
        desc="The first NXT Championship win occurred on April 23, 2024",
        parent=node,
        critical=True,
    )
    claim = f"{_subject_name(ex)} won their first NXT Championship on April 23, 2024."
    await evaluator.verify(
        claim=claim,
        node=win_date_verify_node,
        sources=ex.first_nxt_win.win_urls,
        additional_instruction="Verify the page explicitly states the first NXT Championship victory occurred on April 23, 2024.",
    )

    # URL existence for opponent info
    opponent_url_node = evaluator.add_custom_node(
        result=_has_urls(ex.first_nxt_win.opponent_urls),
        id="Opponent_URL",
        desc="URL reference provided for the opponent information",
        parent=node,
        critical=True,
    )

    # Verify opponent defeated is Ilja Dragunov
    opponent_verify_node = evaluator.add_leaf(
        id="Defeated_Opponent",
        desc="The wrestler defeated Ilja Dragunov to win the first NXT Championship",
        parent=node,
        critical=True,
    )
    claim = f"{_subject_name(ex)} defeated Ilja Dragunov to win their first NXT Championship."
    await evaluator.verify(
        claim=claim,
        node=opponent_verify_node,
        sources=_combine_urls(ex.first_nxt_win.opponent_urls, ex.first_nxt_win.win_urls),
        additional_instruction="Confirm the source clearly connects defeating Ilja Dragunov with winning the first NXT Championship.",
    )


async def build_vengeance_day(evaluator: Evaluator, parent, ex: WrestlerExtraction):
    node = evaluator.add_parallel(
        id="Vengeance_Day_Participation",
        desc="Verification of the wrestler's participation at NXT Vengeance Day 2024",
        parent=parent,
        critical=True,
    )

    # URL existence
    event_url_node = evaluator.add_custom_node(
        result=_has_urls(ex.vengeance_day.event_urls),
        id="Event_URL",
        desc="URL reference provided for NXT Vengeance Day participation",
        parent=node,
        critical=True,
    )

    # Verify event date participation
    event_date_verify_node = evaluator.add_leaf(
        id="Event_Date",
        desc="The wrestler competed at NXT Vengeance Day on February 10, 2024",
        parent=node,
        critical=True,
    )
    claim = f"{_subject_name(ex)} competed at NXT Vengeance Day 2024 on February 10, 2024."
    await evaluator.verify(
        claim=claim,
        node=event_date_verify_node,
        sources=ex.vengeance_day.event_urls,
        additional_instruction="Verify the source explicitly states this wrestler competed at NXT Vengeance Day on Feb 10, 2024.",
    )

    # Verify match type was an NXT Championship match
    match_type_node = evaluator.add_leaf(
        id="Match_Type",
        desc="The match was an NXT Championship match",
        parent=node,
        critical=True,
    )
    claim = f"At NXT Vengeance Day 2024, {_subject_name(ex)} competed in an NXT Championship match."
    await evaluator.verify(
        claim=claim,
        node=match_type_node,
        sources=ex.vengeance_day.event_urls,
        additional_instruction="Confirm the match at Vengeance Day 2024 was for the NXT Championship.",
    )


async def build_total_reigns(evaluator: Evaluator, parent, ex: WrestlerExtraction):
    node = evaluator.add_parallel(
        id="Total_NXT_Reigns",
        desc="Verification of the total number of NXT Championship reigns",
        parent=parent,
        critical=True,
    )

    # URL existence
    reigns_url_node = evaluator.add_custom_node(
        result=_has_urls(ex.nxt_reigns.reigns_urls),
        id="Reigns_URL",
        desc="URL reference provided for championship reigns information",
        parent=node,
        critical=True,
    )

    # Verify total reigns equals two
    reigns_verify_node = evaluator.add_leaf(
        id="Number_of_Reigns",
        desc="The wrestler has had two separate NXT Championship reigns",
        parent=node,
        critical=True,
    )
    claim = f"{_subject_name(ex)} has held the NXT Championship twice."
    await evaluator.verify(
        claim=claim,
        node=reigns_verify_node,
        sources=ex.nxt_reigns.reigns_urls,
        additional_instruction="Verify the source clearly states the wrestler is a two-time NXT Champion.",
    )


async def build_call_up_and_debut(evaluator: Evaluator, parent, ex: WrestlerExtraction):
    main_node = evaluator.add_parallel(
        id="SmackDown_Main_Roster_Career",
        desc="Verification of the wrestler's SmackDown debut and main roster call-up",
        parent=parent,
        critical=True,
    )

    # Call-up details
    callup_node = evaluator.add_parallel(
        id="Call_Up_Details",
        desc="Verification of when the wrestler was called up to the main roster",
        parent=main_node,
        critical=True,
    )

    callup_url_node = evaluator.add_custom_node(
        result=_has_urls(ex.call_up.call_up_urls),
        id="Call_Up_URL",
        desc="URL reference provided for main roster call-up information",
        parent=callup_node,
        critical=True,
    )

    callup_month_node = evaluator.add_leaf(
        id="Call_Up_Month",
        desc="The wrestler was called up to the main roster in December 2025",
        parent=callup_node,
        critical=True,
    )
    claim = f"{_subject_name(ex)} was called up to the WWE main roster in December 2025."
    await evaluator.verify(
        claim=claim,
        node=callup_month_node,
        sources=ex.call_up.call_up_urls,
        additional_instruction="Verify the page explicitly states the main roster call-up occurred in December 2025.",
    )

    brand_assignment_node = evaluator.add_leaf(
        id="Brand_Assignment",
        desc="The wrestler was assigned to SmackDown",
        parent=callup_node,
        critical=True,
    )
    claim = f"Upon call-up, {_subject_name(ex)} was assigned to SmackDown."
    await evaluator.verify(
        claim=claim,
        node=brand_assignment_node,
        sources=ex.call_up.call_up_urls,
        additional_instruction="Verify the source states the brand assignment was SmackDown.",
    )

    # Debut match details
    debut_node = evaluator.add_parallel(
        id="Debut_Match_Details",
        desc="Verification of the wrestler's SmackDown debut match",
        parent=main_node,
        critical=True,
    )

    # Debut Date URL existence
    debut_date_url_node = evaluator.add_custom_node(
        result=_has_urls(ex.smackdown_debut.debut_date_urls),
        id="Debut_Date_URL",
        desc="URL reference provided for debut date",
        parent=debut_node,
        critical=True,
    )
    # Debut date verification
    debut_date_node = evaluator.add_leaf(
        id="Debut_Date",
        desc="The SmackDown debut occurred on January 9, 2026",
        parent=debut_node,
        critical=True,
    )
    claim = f"{_subject_name(ex)} made their SmackDown debut on January 9, 2026."
    await evaluator.verify(
        claim=claim,
        node=debut_date_node,
        sources=ex.smackdown_debut.debut_date_urls,
        additional_instruction="Verify the page explicitly states the debut date as January 9, 2026.",
    )

    # Debut Opponent URL existence
    debut_opponent_url_node = evaluator.add_custom_node(
        result=_has_urls(ex.smackdown_debut.debut_opponent_urls),
        id="Debut_Opponent_URL",
        desc="URL reference provided for debut opponent information",
        parent=debut_node,
        critical=True,
    )
    # Debut opponent verification
    debut_opponent_node = evaluator.add_leaf(
        id="Debut_Opponent",
        desc="The debut match opponent was Rey Fenix",
        parent=debut_node,
        critical=True,
    )
    claim = f"The SmackDown debut opponent for {_subject_name(ex)} was Rey Fenix."
    await evaluator.verify(
        claim=claim,
        node=debut_opponent_node,
        sources=ex.smackdown_debut.debut_opponent_urls,
        additional_instruction="Verify the page lists Rey Fenix as the opponent in the wrestler's SmackDown debut.",
    )

    # Debut Result URL existence
    debut_result_url_node = evaluator.add_custom_node(
        result=_has_urls(ex.smackdown_debut.debut_result_urls),
        id="Result_URL",
        desc="URL reference provided for match result",
        parent=debut_node,
        critical=True,
    )
    # Debut result verification
    debut_result_node = evaluator.add_leaf(
        id="Match_Result",
        desc="The wrestler won their SmackDown debut match",
        parent=debut_node,
        critical=True,
    )
    claim = f"{_subject_name(ex)} won their SmackDown debut match."
    await evaluator.verify(
        claim=claim,
        node=debut_result_node,
        sources=ex.smackdown_debut.debut_result_urls,
        additional_instruction="Verify the page clearly states the wrestler won the SmackDown debut match.",
    )


async def build_tna_history(evaluator: Evaluator, parent, ex: WrestlerExtraction):
    node = evaluator.add_parallel(
        id="TNA_Championship_History",
        desc="Verification of the wrestler's TNA World Championship reign",
        parent=parent,
        critical=True,
    )

    tna_url_node = evaluator.add_custom_node(
        result=_has_urls(ex.tna_info.tna_urls),
        id="TNA_URL",
        desc="URL reference provided for TNA championship information",
        parent=node,
        critical=True,
    )

    tna_verify_node = evaluator.add_leaf(
        id="TNA_Title_Held_2025",
        desc="The wrestler held the TNA World Championship in 2025",
        parent=node,
        critical=True,
    )
    claim = f"{_subject_name(ex)} held the TNA World Championship in 2025."
    await evaluator.verify(
        claim=claim,
        node=tna_verify_node,
        sources=ex.tna_info.tna_urls,
        additional_instruction="Verify the source explicitly states the wrestler held the TNA World Championship in the year 2025.",
    )


async def build_previous_champion_context(evaluator: Evaluator, root, ex: WrestlerExtraction):
    """
    This non-critical context is placed as a separate sibling under the root to satisfy
    the framework constraint: critical parent cannot have non-critical child.
    """
    node = evaluator.add_parallel(
        id="Previous_Champion_Context",
        desc="Information about the NXT Champion who held the title immediately before the wrestler's first win",
        parent=root,
        critical=False,
    )

    # Dragunov win URL existence (critical for this subtree)
    drag_win_url_node = evaluator.add_custom_node(
        result=_has_urls(ex.previous_context.dragunov_win_urls),
        id="Dragunov_Win_URL",
        desc="URL reference provided for Dragunov's championship win",
        parent=node,
        critical=True,
    )

    drag_win_verify_node = evaluator.add_leaf(
        id="Dragunov_Championship_Win",
        desc="Ilja Dragunov won the NXT Championship on September 30, 2023 at NXT No Mercy",
        parent=node,
        critical=False,
    )
    claim = "Ilja Dragunov won the NXT Championship on September 30, 2023 at NXT No Mercy."
    await evaluator.verify(
        claim=claim,
        node=drag_win_verify_node,
        sources=ex.previous_context.dragunov_win_urls,
        additional_instruction="Verify the page explicitly states Dragunov won the NXT Championship at No Mercy on Sept 30, 2023.",
    )

    # Reign end URL existence (critical for this subtree)
    reign_end_url_node = evaluator.add_custom_node(
        result=_has_urls(ex.previous_context.reign_end_urls),
        id="Reign_End_URL",
        desc="URL reference provided for end of Dragunov's reign",
        parent=node,
        critical=True,
    )

    reign_end_verify_node = evaluator.add_leaf(
        id="Dragunov_Reign_End",
        desc="Ilja Dragunov lost the NXT Championship on April 23, 2024",
        parent=node,
        critical=False,
    )
    claim = "Ilja Dragunov lost the NXT Championship on April 23, 2024."
    await evaluator.verify(
        claim=claim,
        node=reign_end_verify_node,
        sources=ex.previous_context.reign_end_urls,
        additional_instruction="Verify the page explicitly states Dragunov lost the NXT Championship on April 23, 2024.",
    )


async def build_correct_wrestler_tree(evaluator: Evaluator, root, ex: WrestlerExtraction):
    """
    Build the critical verification subtree that confirms the wrestler meets all required criteria.
    """
    main = evaluator.add_parallel(
        id="Correct_Wrestler_Identification",
        desc="The answer correctly identifies the WWE wrestler who meets all specified criteria",
        parent=root,
        critical=True,
    )

    # Optional: Name existence gate (critical sibling for downstream verifications)
    if True:
        name_exists_node = evaluator.add_custom_node(
            result=ex.wrestler_name is not None and bool(str(ex.wrestler_name).strip()),
            id="Wrestler_Name_Provided",
            desc="The wrestler's name is provided in the answer",
            parent=main,
            critical=True,
        )

    # NXT Championship history block
    nxt_hist_node = evaluator.add_parallel(
        id="NXT_Championship_History",
        desc="Verification of the wrestler's NXT Championship history and achievements",
        parent=main,
        critical=True,
    )
    await build_first_nxt_win(evaluator, nxt_hist_node, ex)
    await build_vengeance_day(evaluator, nxt_hist_node, ex)
    await build_total_reigns(evaluator, nxt_hist_node, ex)

    # SmackDown main roster career block
    smack_node = evaluator.add_parallel(
        id="SmackDown_Main_Roster_Career",
        desc="Verification of the wrestler's SmackDown debut and main roster call-up",
        parent=main,
        critical=True,
    )
    await build_call_up_and_debut(evaluator, smack_node, ex)

    # TNA championship history block
    tna_node = evaluator.add_parallel(
        id="TNA_Championship_History",
        desc="Verification of the wrestler's TNA World Championship reign",
        parent=main,
        critical=True,
    )
    await build_tna_history(evaluator, tna_node, ex)


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
    Evaluate the agent's answer for the WWE wrestler identification task and return a structured result summary.
    """
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_wrestler(),
        template_class=WrestlerExtraction,
        extraction_name="wrestler_extraction",
    )

    # Build verification tree
    await build_correct_wrestler_tree(evaluator, root, extraction)

    # Non-critical context for previous champion facts (placed as separate sibling under root)
    await build_previous_champion_context(evaluator, root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()