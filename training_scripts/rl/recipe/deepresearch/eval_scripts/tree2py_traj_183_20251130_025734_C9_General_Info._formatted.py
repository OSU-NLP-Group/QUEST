import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broadway_5_shows"
TASK_DESCRIPTION = """
Identify five long-running Broadway musicals that are currently still playing and provide, for each, the show name, current Broadway theater, Broadway opening date, and a supporting reference URL; additionally satisfy each show-specific constraint (revival/opens/moves/largest-theater).
"""

# Optional ground truth expectation (for context only; not used for verification)
EXPECTED_SHOWS = [
    "Chicago (1996 revival)",
    "The Lion King",
    "Wicked",
    "The Book of Mormon",
    "Hamilton",
]
EXPECTED_KEY_FACTS = {
    "Chicago": {
        "opening_date": "November 14, 1996",
        "notes": ["Revival", "Longest-running American musical"]
    },
    "The Lion King": {
        "opening_date": "November 13, 1997",
        "original_theater": "New Amsterdam Theatre",
        "move_date": "June 13, 2006",
        "current_theater": "Minskoff Theatre",
    },
    "Wicked": {
        "opening_date": "October 30, 2003",
        "current_theater": "Gershwin Theatre",
        "capacity_approx": "1,933",
        "largest_broadway_theater": True
    },
    "The Book of Mormon": {
        "opening_date": "March 24, 2011",
        "current_theater": "Eugene O'Neill Theatre",
    },
    "Hamilton": {
        "opening_date": "August 6, 2015",
        "current_theater": "Richard Rodgers Theatre",
    }
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ShowInfo(BaseModel):
    name: Optional[str] = None
    opening_date: Optional[str] = None
    current_theater: Optional[str] = None
    original_theater: Optional[str] = None
    move_date: Optional[str] = None
    theater_capacity: Optional[str] = None  # Keep as string for flexible formats (e.g., "1,933", "~1933")
    is_revival: Optional[bool] = None
    longest_running_american_musical: Optional[bool] = None
    largest_broadway_theater_claim: Optional[bool] = None
    currently_playing: Optional[bool] = None
    reference_urls: List[str] = Field(default_factory=list)


class BroadwayShowsExtraction(BaseModel):
    show1: Optional[ShowInfo] = None  # Chicago (1996 revival)
    show2: Optional[ShowInfo] = None  # The Lion King
    show3: Optional[ShowInfo] = None  # Wicked
    show4: Optional[ShowInfo] = None  # The Book of Mormon
    show5: Optional[ShowInfo] = None  # Hamilton


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_broadway_shows() -> str:
    return """
    Extract the five Broadway musicals mentioned in the answer that satisfy the following specific constraints. For each show, return the requested fields exactly as stated in the answer. Include at least one reference URL per show when available.

    You must fill the following structured JSON object with fields for each show:

    show1 (Chicago (1996 revival)):
      - name
      - opening_date (e.g., "November 14, 1996")
      - current_theater
      - is_revival (true/false, if the answer explicitly states it is a Broadway revival)
      - longest_running_american_musical (true/false, if the answer explicitly states it is the longest-running American musical currently on Broadway)
      - currently_playing (true/false, as explicitly stated)
      - reference_urls (list of URLs cited for Chicago)

    show2 (The Lion King):
      - name
      - opening_date (e.g., "November 13, 1997")
      - original_theater (e.g., "New Amsterdam Theatre")
      - move_date (e.g., "June 13, 2006")
      - current_theater (e.g., "Minskoff Theatre")
      - currently_playing (true/false, as explicitly stated)
      - reference_urls (list of URLs cited for The Lion King)

    show3 (Wicked):
      - name
      - opening_date (e.g., "October 30, 2003")
      - current_theater (e.g., "Gershwin Theatre")
      - theater_capacity (e.g., "1,933" or "~1,933")
      - largest_broadway_theater_claim (true/false, if the answer explicitly claims this theater is the largest on Broadway by seating capacity)
      - currently_playing (true/false, as explicitly stated)
      - reference_urls (list of URLs cited for Wicked)

    show4 (The Book of Mormon):
      - name
      - opening_date (e.g., "March 24, 2011")
      - current_theater (e.g., "Eugene O'Neill Theatre")
      - currently_playing (true/false, as explicitly stated)
      - reference_urls (list of URLs cited for The Book of Mormon)

    show5 (Hamilton):
      - name
      - opening_date (e.g., "August 6, 2015")
      - current_theater (e.g., "Richard Rodgers Theatre")
      - currently_playing (true/false, as explicitly stated)
      - reference_urls (list of URLs cited for Hamilton)

    RULES:
    - Extract only information that is explicitly present in the answer text.
    - For dates, capture the full textual format (e.g., "November 14, 1996").
    - For theater names, preserve exact spelling and punctuation as in the answer (e.g., "Eugene O'Neill Theatre").
    - For boolean fields, set true only if the answer clearly and explicitly states the fact; otherwise false or null.
    - For reference_urls, extract only valid URL strings mentioned in the answer (including markdown links).
    - If a field is not mentioned, return null for that field or an empty list where appropriate.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_show_1_chicago(evaluator: Evaluator, parent_node, show: Optional[ShowInfo]) -> None:
    node = evaluator.add_parallel(
        id="Show_1_Chicago",
        desc="First show: longest-running American musical currently on Broadway; opened as a revival in November 1996; provide exact opening date, current theater, and references.",
        parent=parent_node,
        critical=False
    )

    # Chicago_Name
    chicago_name = evaluator.add_leaf(
        id="Chicago_Name",
        desc="Show is identified as Chicago (1996 Broadway revival).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The first show is identified as 'Chicago' (the 1996 Broadway revival).",
        node=chicago_name,
        additional_instruction="Verify the answer text mentions Chicago and indicates the 1996 revival; minor wording variations are acceptable."
    )

    # Chicago_Is_Revival
    chicago_revival = evaluator.add_leaf(
        id="Chicago_Is_Revival",
        desc="Show is explicitly identified/described as a Broadway revival production.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the Chicago production is a Broadway revival.",
        node=chicago_revival,
        additional_instruction="Check the answer text for explicit mention of 'revival' for Chicago."
    )

    # Chicago_Opening_Date
    chicago_opening = evaluator.add_leaf(
        id="Chicago_Opening_Date",
        desc="Broadway opening date is stated as November 14, 1996.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that Chicago opened on Broadway on November 14, 1996.",
        node=chicago_opening,
        additional_instruction="Allow minor formatting variations (e.g., 'Nov 14, 1996')."
    )

    # Chicago_Longest_Running_American_Musical
    chicago_longest = evaluator.add_leaf(
        id="Chicago_Longest_Running_American_Musical",
        desc="Show is identified as the longest-running American musical currently on Broadway.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that Chicago is the longest-running American musical currently on Broadway.",
        node=chicago_longest,
        additional_instruction="Focus on the phrasing 'longest-running American musical' and 'currently on Broadway'."
    )

    # Chicago_Current_Theater_Provided_And_Correct
    chicago_theater = evaluator.add_leaf(
        id="Chicago_Current_Theater_Provided_And_Correct",
        desc="The current Broadway theater where Chicago plays is provided and is correct.",
        parent=node,
        critical=True
    )
    current_theater_text = (show.current_theater or "").strip() if show else ""
    await evaluator.verify(
        claim=f"Chicago currently plays at the {current_theater_text}.",
        node=chicago_theater,
        sources=show.reference_urls if show else [],
        additional_instruction="Verify that the provided sources confirm the current theater for Chicago. Accept minor name variations (Theatre vs Theater)."
    )

    # Chicago_Reference_URL_Provided
    chicago_refs_provided = evaluator.add_custom_node(
        result=bool(show and show.reference_urls and len(show.reference_urls) > 0),
        id="Chicago_Reference_URL_Provided",
        desc="At least one reference URL is provided for Chicago.",
        parent=node,
        critical=True
    )

    # Chicago_Reference_Corroborates_Facts
    chicago_refs_corroborate = evaluator.add_leaf(
        id="Chicago_Reference_Corroborates_Facts",
        desc="Provided reference(s) corroborate the required Chicago facts (opening date, revival status, longest-running status, and current theater).",
        parent=node,
        critical=True
    )
    combined_claim = (
        f"Chicago opened on Broadway on November 14, 1996 as a revival; "
        f"it is the longest-running American musical currently on Broadway; "
        f"and it currently plays at the {current_theater_text}."
    )
    await evaluator.verify(
        claim=combined_claim,
        node=chicago_refs_corroborate,
        sources=show.reference_urls if show else [],
        extra_prerequisites=[chicago_refs_provided],
        additional_instruction="It is acceptable if different referenced pages corroborate different parts of the claim collectively."
    )


async def verify_show_2_lion_king(evaluator: Evaluator, parent_node, show: Optional[ShowInfo]) -> None:
    node = evaluator.add_parallel(
        id="Show_2_Lion_King",
        desc="Second show: opened in Nov 1997; originally at New Amsterdam Theatre; moved in June 2006; currently at the Minskoff Theatre; provide dates, theaters, and references.",
        parent=parent_node,
        critical=False
    )

    # Name
    name_leaf = evaluator.add_leaf(
        id="LionKing_Name",
        desc="Show is identified as The Lion King.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The second show is identified as 'The Lion King'.",
        node=name_leaf,
        additional_instruction="Minor variants like missing 'The' are acceptable."
    )

    # Opening Date
    opening_leaf = evaluator.add_leaf(
        id="LionKing_Opening_Date",
        desc="Broadway opening date is stated as November 13, 1997.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that The Lion King opened on Broadway on November 13, 1997.",
        node=opening_leaf,
        additional_instruction="Allow 'Nov 13, 1997'."
    )

    # Original Theater
    original_leaf = evaluator.add_leaf(
        id="LionKing_Original_Theater",
        desc="Original opening theater is stated as the New Amsterdam Theatre.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that The Lion King originally opened at the New Amsterdam Theatre.",
        node=original_leaf,
        additional_instruction="Ensure the theater name matches 'New Amsterdam Theatre'."
    )

    # Move Date
    move_leaf = evaluator.add_leaf(
        id="LionKing_Move_Date",
        desc="Theater move date is stated as June 13, 2006.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that The Lion King moved theaters on June 13, 2006.",
        node=move_leaf,
        additional_instruction="The destination theater move is to the Minskoff Theatre."
    )

    # Current Theater
    current_leaf = evaluator.add_leaf(
        id="LionKing_Current_Theater",
        desc="Current theater is stated as the Minskoff Theatre.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that The Lion King currently plays at the Minskoff Theatre.",
        node=current_leaf,
        additional_instruction="Allow minor variants in spelling or 'Theatre' vs 'Theater'."
    )

    # Currently Playing
    playing_leaf = evaluator.add_leaf(
        id="LionKing_Currently_Playing",
        desc="Show is identified as currently still playing on Broadway.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer indicates that The Lion King is currently playing on Broadway.",
        node=playing_leaf,
        additional_instruction="Look for language implying ongoing performances."
    )

    # References provided
    refs_provided = evaluator.add_custom_node(
        result=bool(show and show.reference_urls and len(show.reference_urls) > 0),
        id="LionKing_Reference_URL_Provided",
        desc="At least one reference URL is provided for The Lion King.",
        parent=node,
        critical=True
    )

    # References corroborate facts
    refs_corroborate = evaluator.add_leaf(
        id="LionKing_Reference_Corroborates_Facts",
        desc="Provided reference(s) corroborate the required Lion King facts (opening date, original theater, move date, and current theater).",
        parent=node,
        critical=True
    )
    claim = (
        "The Lion King opened on November 13, 1997 at the New Amsterdam Theatre, "
        "moved to the Minskoff Theatre on June 13, 2006, and currently plays at the Minskoff Theatre."
    )
    await evaluator.verify(
        claim=claim,
        node=refs_corroborate,
        sources=show.reference_urls if show else [],
        extra_prerequisites=[refs_provided],
        additional_instruction="Different pages may corroborate different parts of the claim; collectively they should confirm these facts."
    )


async def verify_show_3_wicked(evaluator: Evaluator, parent_node, show: Optional[ShowInfo]) -> None:
    node = evaluator.add_parallel(
        id="Show_3_Wicked",
        desc="Third show: opened in Oct 2003; plays at Broadway’s largest theater (Gershwin); provide opening date, theater name, capacity, and references.",
        parent=parent_node,
        critical=False
    )

    # Name
    name_leaf = evaluator.add_leaf(
        id="Wicked_Name",
        desc="Show is identified as Wicked.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The third show is identified as 'Wicked'.",
        node=name_leaf,
        additional_instruction="Exact match or clear identification of 'Wicked'."
    )

    # Opening Date
    opening_leaf = evaluator.add_leaf(
        id="Wicked_Opening_Date",
        desc="Broadway opening date is stated as October 30, 2003.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that Wicked opened on Broadway on October 30, 2003.",
        node=opening_leaf,
        additional_instruction="Minor format variations acceptable."
    )

    # Current Theater
    theater_leaf = evaluator.add_leaf(
        id="Wicked_Current_Theater",
        desc="Current theater is stated as the Gershwin Theatre.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that Wicked currently plays at the Gershwin Theatre.",
        node=theater_leaf,
        additional_instruction="Allow 'Theatre' vs 'Theater'."
    )

    # Gershwin Seating Capacity (~1,933)
    capacity_leaf = evaluator.add_leaf(
        id="Gershwin_Seating_Capacity",
        desc="Gershwin Theatre seating capacity is stated as approximately 1,933 seats.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the Gershwin Theatre has a seating capacity of approximately 1,933 seats.",
        node=capacity_leaf,
        additional_instruction="Approximate phrasing is acceptable (e.g., ~1,933, around 1,933)."
    )

    # Gershwin is Largest Broadway Theater
    largest_leaf = evaluator.add_leaf(
        id="Gershwin_Is_Largest_Broadway_Theater",
        desc="Gershwin Theatre is identified/confirmed as the largest Broadway theater by seating capacity.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the Gershwin Theatre is the largest Broadway theater by seating capacity.",
        node=largest_leaf,
        additional_instruction="Focus on 'largest by seating capacity' phrasing."
    )

    # Currently Playing
    playing_leaf = evaluator.add_leaf(
        id="Wicked_Currently_Playing",
        desc="Show is identified as currently still playing on Broadway.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer indicates that Wicked is currently playing on Broadway.",
        node=playing_leaf,
        additional_instruction="Look for 'currently' or ticketing language."
    )

    # References provided
    refs_provided = evaluator.add_custom_node(
        result=bool(show and show.reference_urls and len(show.reference_urls) > 0),
        id="Wicked_Reference_URL_Provided",
        desc="At least one reference URL is provided for Wicked.",
        parent=node,
        critical=True
    )

    # References corroborate facts
    refs_corroborate = evaluator.add_leaf(
        id="Wicked_Reference_Corroborates_Facts",
        desc="Provided reference(s) corroborate the required Wicked facts (opening date, current theater, capacity, and largest-theater claim).",
        parent=node,
        critical=True
    )
    cap_text = (show.theater_capacity or "1,933")
    claim = (
        f"Wicked opened on October 30, 2003; it plays at the Gershwin Theatre; "
        f"the Gershwin Theatre has approximately {cap_text} seats and is the largest Broadway theater by seating capacity."
    )
    await evaluator.verify(
        claim=claim,
        node=refs_corroborate,
        sources=show.reference_urls if show else [],
        extra_prerequisites=[refs_provided],
        additional_instruction="Collective corroboration across the provided references is acceptable."
    )


async def verify_show_4_mormon(evaluator: Evaluator, parent_node, show: Optional[ShowInfo]) -> None:
    node = evaluator.add_parallel(
        id="Show_4_Book_of_Mormon",
        desc="Fourth show: opened in March 2011 at the Eugene O'Neill Theatre; provide exact opening date, confirm theater, and references.",
        parent=parent_node,
        critical=False
    )

    # Name
    name_leaf = evaluator.add_leaf(
        id="Mormon_Name",
        desc="Show is identified as The Book of Mormon.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The fourth show is identified as 'The Book of Mormon'.",
        node=name_leaf,
        additional_instruction="Exact match or clear identification."
    )

    # Opening Date
    opening_leaf = evaluator.add_leaf(
        id="Mormon_Opening_Date",
        desc="Broadway opening date is stated as March 24, 2011.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that The Book of Mormon opened on Broadway on March 24, 2011.",
        node=opening_leaf,
        additional_instruction="Minor format variations acceptable."
    )

    # Current Theater
    theater_leaf = evaluator.add_leaf(
        id="Mormon_Current_Theater",
        desc="Current theater is stated as the Eugene O'Neill Theatre.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that The Book of Mormon plays at the Eugene O'Neill Theatre.",
        node=theater_leaf,
        additional_instruction="Respect apostrophe and spelling."
    )

    # Currently Playing
    playing_leaf = evaluator.add_leaf(
        id="Mormon_Currently_Playing",
        desc="Show is identified as currently still playing on Broadway.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer indicates that The Book of Mormon is currently playing on Broadway.",
        node=playing_leaf,
        additional_instruction="Look for ongoing performance language."
    )

    # References provided
    refs_provided = evaluator.add_custom_node(
        result=bool(show and show.reference_urls and len(show.reference_urls) > 0),
        id="Mormon_Reference_URL_Provided",
        desc="At least one reference URL is provided for The Book of Mormon.",
        parent=node,
        critical=True
    )

    # References corroborate facts
    refs_corroborate = evaluator.add_leaf(
        id="Mormon_Reference_Corroborates_Facts",
        desc="Provided reference(s) corroborate the required Book of Mormon facts (opening date and Eugene O'Neill Theatre).",
        parent=node,
        critical=True
    )
    claim = (
        "The Book of Mormon opened on March 24, 2011 and plays at the Eugene O'Neill Theatre."
    )
    await evaluator.verify(
        claim=claim,
        node=refs_corroborate,
        sources=show.reference_urls if show else [],
        extra_prerequisites=[refs_provided],
        additional_instruction="Collective corroboration across references is acceptable."
    )


async def verify_show_5_hamilton(evaluator: Evaluator, parent_node, show: Optional[ShowInfo]) -> None:
    node = evaluator.add_parallel(
        id="Show_5_Hamilton",
        desc="Fifth show: opened in August 2015 at the Richard Rodgers Theatre; provide exact opening date, confirm theater, and references.",
        parent=parent_node,
        critical=False
    )

    # Name
    name_leaf = evaluator.add_leaf(
        id="Hamilton_Name",
        desc="Show is identified as Hamilton.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The fifth show is identified as 'Hamilton'.",
        node=name_leaf,
        additional_instruction="Exact match or clear identification."
    )

    # Opening Date
    opening_leaf = evaluator.add_leaf(
        id="Hamilton_Opening_Date",
        desc="Broadway opening date is stated as August 6, 2015.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that Hamilton opened on Broadway on August 6, 2015.",
        node=opening_leaf,
        additional_instruction="Minor format variations acceptable."
    )

    # Current Theater
    theater_leaf = evaluator.add_leaf(
        id="Hamilton_Current_Theater",
        desc="Current theater is stated as the Richard Rodgers Theatre.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that Hamilton plays at the Richard Rodgers Theatre.",
        node=theater_leaf,
        additional_instruction="Allow minor spelling variants."
    )

    # Currently Playing
    playing_leaf = evaluator.add_leaf(
        id="Hamilton_Currently_Playing",
        desc="Show is identified as currently still playing on Broadway.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer indicates that Hamilton is currently playing on Broadway.",
        node=playing_leaf,
        additional_instruction="Look for phrasing indicating ongoing performances."
    )

    # References provided
    refs_provided = evaluator.add_custom_node(
        result=bool(show and show.reference_urls and len(show.reference_urls) > 0),
        id="Hamilton_Reference_URL_Provided",
        desc="At least one reference URL is provided for Hamilton.",
        parent=node,
        critical=True
    )

    # References corroborate facts
    refs_corroborate = evaluator.add_leaf(
        id="Hamilton_Reference_Corroborates_Facts",
        desc="Provided reference(s) corroborate the required Hamilton facts (opening date and Richard Rodgers Theatre).",
        parent=node,
        critical=True
    )
    claim = (
        "Hamilton opened on August 6, 2015 and plays at the Richard Rodgers Theatre."
    )
    await evaluator.verify(
        claim=claim,
        node=refs_corroborate,
        sources=show.reference_urls if show else [],
        extra_prerequisites=[refs_provided],
        additional_instruction="Collective corroboration across references is acceptable."
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
    Evaluate an answer for the Broadway long-running musicals task.
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
        default_model=model
    )

    # Record expected info as ground truth context (for auditing)
    evaluator.add_ground_truth({
        "expected_shows": EXPECTED_SHOWS,
        "key_facts": EXPECTED_KEY_FACTS
    }, gt_type="expected_context")

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_broadway_shows(),
        template_class=BroadwayShowsExtraction,
        extraction_name="broadway_shows_extraction"
    )

    # Build verification tree per show
    await verify_show_1_chicago(evaluator, root, extracted.show1)
    await verify_show_2_lion_king(evaluator, root, extracted.show2)
    await verify_show_3_wicked(evaluator, root, extracted.show3)
    await verify_show_4_mormon(evaluator, root, extracted.show4)
    await verify_show_5_hamilton(evaluator, root, extracted.show5)

    # Return structured summary
    return evaluator.get_summary()