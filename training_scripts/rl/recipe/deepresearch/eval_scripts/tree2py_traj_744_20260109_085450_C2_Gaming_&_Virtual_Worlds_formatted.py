import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "vr_studio_meta_award_2024"
TASK_DESCRIPTION = """
What is the name of the VR game studio that meets all of the following criteria: (1) The studio developed a game that won the Best VR/AR Game award at The Game Awards 2024, (2) The studio was acquired by Meta (formerly Facebook) between January 2021 and December 2022, inclusive, (3) The studio is headquartered in a U.S. state located west of the Mississippi River, (4) The award-winning game was released in the fourth quarter of 2024 (October through December), and (5) The studio's founder previously worked at major gaming companies before founding this studio? Provide the studio name and include reference URLs that verify each of the above criteria.
"""

# States typically considered west of the Mississippi River (includes states entirely or primarily west of the river)
WEST_OF_MISSISSIPPI_STATES = {
    "alaska", "hawaii", "washington", "oregon", "california", "nevada", "idaho", "montana",
    "wyoming", "utah", "arizona", "new mexico", "colorado", "north dakota", "south dakota",
    "nebraska", "kansas", "oklahoma", "texas", "minnesota", "iowa", "missouri", "arkansas", "louisiana"
}
STATE_ABBREV_MAP = {
    "AK": "alaska", "HI": "hawaii", "WA": "washington", "OR": "oregon", "CA": "california", "NV": "nevada", "ID": "idaho",
    "MT": "montana", "WY": "wyoming", "UT": "utah", "AZ": "arizona", "NM": "new mexico", "CO": "colorado", "ND": "north dakota",
    "SD": "south dakota", "NE": "nebraska", "KS": "kansas", "OK": "oklahoma", "TX": "texas", "MN": "minnesota", "IA": "iowa",
    "MO": "missouri", "AR": "arkansas", "LA": "louisiana"
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StudioExtraction(BaseModel):
    """Structured information extracted from the agent's answer for this task."""
    studio_name: Optional[str] = None

    # Award-winning game and development claim
    award_game_title: Optional[str] = None
    award_win_claimed: Optional[bool] = False
    developer_claimed: Optional[bool] = False
    award_urls: List[str] = Field(default_factory=list)

    # Release date in Q4 2024
    release_date_text: Optional[str] = None
    release_urls: List[str] = Field(default_factory=list)

    # Acquisition by Meta within the timeframe
    acquired_by_meta_claimed: Optional[bool] = False
    acquisition_date_text: Optional[str] = None
    acquisition_urls: List[str] = Field(default_factory=list)

    # Headquarters location (city/state) and source
    headquarters_city: Optional[str] = None
    headquarters_state: Optional[str] = None
    hq_urls: List[str] = Field(default_factory=list)

    # Founder background (worked at major gaming companies) and source
    founder_name: Optional[str] = None
    founder_prior_employers: List[str] = Field(default_factory=list)
    founder_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_studio_task() -> str:
    return """
    Extract the following information from the answer about the identified VR game studio and its award-winning game. If any item is not explicitly present in the answer, return null (for string) or false (for boolean) or an empty list (for URLs).

    Required fields:
    1. studio_name: The explicit name of the VR game studio that the answer claims matches all criteria.
    2. award_game_title: The title of the game the studio developed that the answer claims won Best VR/AR Game at The Game Awards 2024.
    3. award_win_claimed: Boolean. True if the answer explicitly states that the identified game won "Best VR/AR Game" at "The Game Awards 2024"; otherwise False.
    4. developer_claimed: Boolean. True if the answer explicitly states that the identified studio developed the award-winning game; otherwise False.
    5. award_urls: A list of URL(s) provided in the answer that support the claim that the game won Best VR/AR Game at The Game Awards 2024. Extract only URLs explicitly present in the answer.
    6. release_date_text: The release date or release window provided for the award-winning game (e.g., "November 2024", "Dec 12, 2024", "Q4 2024").
    7. release_urls: A list of URL(s) provided that support the release date/window for the game.
    8. acquired_by_meta_claimed: Boolean. True if the answer explicitly states the studio was acquired by Meta (formerly Facebook); otherwise False.
    9. acquisition_date_text: The acquisition date, month-year, or year provided in the answer (must be between January 2021 and December 2022 to satisfy the criterion).
    10. acquisition_urls: A list of URL(s) provided that support Meta's acquisition of the studio and the stated acquisition date/timeframe.
    11. headquarters_city: The headquarters city of the studio (if provided).
    12. headquarters_state: The headquarters state of the studio (e.g., "California", "NV"). If only the city is provided, but the state is inferable from the answer, include the state explicitly; otherwise return null.
    13. hq_urls: A list of URL(s) provided that support the stated headquarters location.
    14. founder_name: The founder's name (primary founder) explicitly mentioned in the answer.
    15. founder_prior_employers: A list of company names that the answer claims the founder previously worked at (should be major gaming companies like EA, Blizzard, Riot, Ubisoft, PlayStation Studios, etc.).
    16. founder_urls: A list of URL(s) provided that support the founder's prior work experience.

    SPECIAL RULES FOR URL SOURCES EXTRACTION:
    - Only extract URLs explicitly present in the answer (plain URLs or markdown links). Do not invent URLs.
    - If the answer references a site without an explicit URL, return an empty list for that sources field.

    Return the JSON object with exactly these keys.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    normalized = []
    for u in urls:
        if isinstance(u, str) and u.strip():
            normalized.append(u.strip())
    return normalized


def _state_to_key(state: Optional[str]) -> Optional[str]:
    if not _non_empty(state):
        return None
    s = state.strip()
    if len(s) == 2 and s.upper() in STATE_ABBREV_MAP:
        return STATE_ABBREV_MAP[s.upper()]
    return s.lower()


def is_state_west_of_mississippi(state: Optional[str]) -> bool:
    key = _state_to_key(state)
    return bool(key and key in WEST_OF_MISSISSIPPI_STATES)


def contains_year(text: Optional[str], year: int) -> bool:
    if not _non_empty(text):
        return False
    return str(year) in text


def acquisition_in_2021_2022(acq_text: Optional[str]) -> bool:
    if not _non_empty(acq_text):
        return False
    years = re.findall(r"\b(20\d{2})\b", acq_text)
    # Accept if any referenced year is 2021 or 2022
    return any(y in {"2021", "2022"} for y in years)


def is_q4_2024(release_text: Optional[str]) -> bool:
    """
    Returns True if the provided release window/date is between Oct 1 and Dec 31, 2024 inclusive.
    Accepts patterns like "October 2024", "Nov 12, 2024", "2024-12-01", "Q4 2024", "Fourth quarter 2024".
    """
    if not _non_empty(release_text):
        return False
    t = release_text.lower().strip()

    # Direct Q4 mention
    if "q4 2024" in t or "fourth quarter 2024" in t or "4th quarter 2024" in t:
        return True

    # Month name checks
    months = {
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }
    for m_name, m_num in months.items():
        if m_name in t and "2024" in t:
            return True

    # Numeric patterns
    # e.g., 2024-10-xx or 2024/10/xx or 10/2024 or 11/2024 or 12/2024
    if re.search(r"2024[-/](10|11|12)\b", t):
        return True
    if re.search(r"\b(10|11|12)[-/]2024\b", t):
        return True

    return False


def _build_city_state_str(city: Optional[str], state: Optional[str]) -> str:
    if _non_empty(city) and _non_empty(state):
        return f"{city.strip()}, {state.strip()}"
    elif _non_empty(state):
        return state.strip()
    elif _non_empty(city):
        return city.strip()
    else:
        return ""


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def _verify_with_urls_if_available(
    evaluator: Evaluator,
    claim: str,
    node,
    urls: Optional[List[str]],
    additional_instruction: str
):
    urls = _normalize_urls(urls)
    if urls:
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=additional_instruction
        )
    else:
        # No URLs provided → this leaf must fail because it requires a verifying URL
        node.score = 0.0
        node.status = "failed"


async def build_verification_tree(evaluator: Evaluator, root, info: StudioExtraction) -> None:
    """
    Build the verification tree and execute necessary verifications based on extracted info.
    """
    # Top-level critical node aggregating all criteria
    top = evaluator.add_parallel(
        id="Correct_Studio_Identified",
        desc="Answer identifies a VR game studio that satisfies all stated constraints and provides reference URLs verifying each criterion.",
        parent=root,
        critical=True
    )

    # 1) Studio name provided (critical)
    evaluator.add_custom_node(
        result=_non_empty(info.studio_name),
        id="Studio_Name_Provided",
        desc="The answer explicitly provides the studio name.",
        parent=top,
        critical=True
    )

    # 2) Award criterion (critical, parallel)
    award_node = evaluator.add_parallel(
        id="Award_Criterion_Verification",
        desc="The studio developed a game that won Best VR/AR Game at The Game Awards 2024, with a verifying reference URL.",
        parent=top,
        critical=True
    )

    # 2.1 Award Winner Fact (critical)
    evaluator.add_custom_node(
        result=_non_empty(info.award_game_title) and bool(info.award_win_claimed) and bool(info.developer_claimed),
        id="Award_Winner_Fact",
        desc="The answer identifies the award-winning game and states it won Best VR/AR Game at The Game Awards 2024 (and that the studio developed it).",
        parent=award_node,
        critical=True
    )

    # 2.2 Award Winner Source URL (critical, verify by urls)
    award_src_leaf = evaluator.add_leaf(
        id="Award_Winner_Source_URL",
        desc="At least one reference URL is provided that supports the claim that the identified game won Best VR/AR Game at The Game Awards 2024.",
        parent=award_node,
        critical=True
    )
    game_title = info.award_game_title or ""
    await _verify_with_urls_if_available(
        evaluator,
        claim=f"The game '{game_title}' won Best VR/AR Game at The Game Awards 2024.",
        node=award_src_leaf,
        urls=info.award_urls,
        additional_instruction="Verify that the provided page(s) explicitly state that the identified game won (not just nominated) Best VR/AR Game at The Game Awards 2024."
    )

    # 3) Release date criterion (critical, parallel)
    release_node = evaluator.add_parallel(
        id="Release_Date_Criterion_Verification",
        desc="The award-winning game’s release date is in Q4 2024 (Oct–Dec 2024), with a verifying reference URL.",
        parent=top,
        critical=True
    )

    # 3.1 Q4 2024 Release Fact (critical)
    evaluator.add_custom_node(
        result=_non_empty(info.release_date_text) and is_q4_2024(info.release_date_text),
        id="Q4_2024_Release_Fact",
        desc="The answer provides the award-winning game’s release date (or release window) and it falls between October 1 and December 31, 2024 (inclusive).",
        parent=release_node,
        critical=True
    )

    # 3.2 Release Date Source URL (critical)
    release_src_leaf = evaluator.add_leaf(
        id="Release_Date_Source_URL",
        desc="At least one reference URL is provided that supports the stated release date/window being in Q4 2024.",
        parent=release_node,
        critical=True
    )
    release_text = info.release_date_text or "Q4 2024"
    await _verify_with_urls_if_available(
        evaluator,
        claim=f"The game '{game_title}' was released in {release_text}.",
        node=release_src_leaf,
        urls=info.release_urls,
        additional_instruction="Verify the game's release date or release window from the provided source(s). Accept official store pages, press releases, or reputable media coverage."
    )

    # 4) Acquisition criterion (critical, parallel)
    acq_node = evaluator.add_parallel(
        id="Acquisition_Criterion_Verification",
        desc="The studio was acquired by Meta (formerly Facebook) between Jan 2021 and Dec 2022 (inclusive), with a verifying reference URL.",
        parent=top,
        critical=True
    )

    # 4.1 Meta Acquisition In Timeframe Fact (critical)
    evaluator.add_custom_node(
        result=bool(info.acquired_by_meta_claimed) and _non_empty(info.acquisition_date_text) and acquisition_in_2021_2022(info.acquisition_date_text),
        id="Meta_Acquisition_In_Timeframe_Fact",
        desc="The answer states the studio was acquired by Meta (formerly Facebook) and that the acquisition occurred between January 1, 2021 and December 31, 2022 (inclusive).",
        parent=acq_node,
        critical=True
    )

    # 4.2 Acquisition Source URL (critical)
    acq_src_leaf = evaluator.add_leaf(
        id="Acquisition_Source_URL",
        desc="At least one reference URL is provided that supports Meta’s acquisition of the studio and the acquisition date/year falling within the specified timeframe.",
        parent=acq_node,
        critical=True
    )
    acq_text = info.acquisition_date_text or "2021/2022"
    studio_name = info.studio_name or ""
    await _verify_with_urls_if_available(
        evaluator,
        claim=f"Meta (formerly Facebook) acquired {studio_name} in {acq_text}.",
        node=acq_src_leaf,
        urls=info.acquisition_urls,
        additional_instruction="Verify that the studio was acquired by Meta and that the acquisition date/year is in 2021 or 2022."
    )

    # 5) Headquarters location criterion (critical, parallel)
    hq_node = evaluator.add_parallel(
        id="Headquarters_Location_Criterion_Verification",
        desc="The studio is headquartered in a U.S. state west of the Mississippi River, with a verifying reference URL.",
        parent=top,
        critical=True
    )

    # 5.1 Western US HQ Fact (critical)
    evaluator.add_custom_node(
        result=_non_empty(info.headquarters_state) and is_state_west_of_mississippi(info.headquarters_state),
        id="Western_US_HQ_Fact",
        desc="The answer states the studio’s headquarters state and that it is west of the Mississippi River.",
        parent=hq_node,
        critical=True
    )

    # 5.2 HQ Location Source URL (critical)
    hq_src_leaf = evaluator.add_leaf(
        id="HQ_Location_Source_URL",
        desc="At least one reference URL is provided that supports the stated headquarters location (sufficient to verify it is in a western U.S. state).",
        parent=hq_node,
        critical=True
    )
    city_state_str = _build_city_state_str(info.headquarters_city, info.headquarters_state)
    location_claim = city_state_str if city_state_str else (info.headquarters_state or "")
    await _verify_with_urls_if_available(
        evaluator,
        claim=f"{studio_name} is headquartered in {location_claim}.",
        node=hq_src_leaf,
        urls=info.hq_urls,
        additional_instruction="Verify the studio's headquarters location (city/state) from the provided source(s)."
    )

    # 6) Founder background criterion (critical, parallel)
    founder_node = evaluator.add_parallel(
        id="Founder_Background_Criterion_Verification",
        desc="The studio’s founder previously worked at major gaming companies before founding the studio, with a verifying reference URL.",
        parent=top,
        critical=True
    )

    # 6.1 Founder Background Fact (critical)
    evaluator.add_custom_node(
        result=_non_empty(info.founder_name) and len(info.founder_prior_employers) > 0,
        id="Founder_Background_Fact",
        desc="The answer identifies the founder and states at least one prior role/employer at a major gaming company before founding the studio.",
        parent=founder_node,
        critical=True
    )

    # 6.2 Founder Background Source URL (critical)
    founder_src_leaf = evaluator.add_leaf(
        id="Founder_Background_Source_URL",
        desc="At least one reference URL is provided that supports the founder’s prior work experience at the cited major gaming company(ies) before founding the studio.",
        parent=founder_node,
        critical=True
    )
    employers_str = ", ".join(info.founder_prior_employers) if info.founder_prior_employers else "major gaming companies"
    founder_name = info.founder_name or ""
    await _verify_with_urls_if_available(
        evaluator,
        claim=f"Before founding {studio_name}, {founder_name} worked at {employers_str}.",
        node=founder_src_leaf,
        urls=info.founder_urls,
        additional_instruction="Verify that the founder previously worked at the named major gaming companies prior to founding the studio."
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
    Evaluate an answer for the VR game studio criteria task.
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

    # Extract structured info from the answer
    info: StudioExtraction = await evaluator.extract(
        prompt=prompt_extract_studio_task(),
        template_class=StudioExtraction,
        extraction_name="studio_extraction"
    )

    # Add custom info useful for understanding geography/timeframe checks
    evaluator.add_custom_info(
        {
            "west_of_mississippi_states": sorted(list(WEST_OF_MISSISSIPPI_STATES)),
            "acquisition_timeframe_years": [2021, 2022],
            "release_window_required": "Q4 2024 (Oct–Dec 2024)"
        },
        info_type="constraints",
        info_name="constraints_used_in_checks"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, info)

    # Return structured summary
    return evaluator.get_summary()