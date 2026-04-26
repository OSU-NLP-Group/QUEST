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
TASK_ID = "aera_2026_posters"
TASK_DESCRIPTION = """
The American Educational Research Association (AERA) is holding its 2026 Annual Meeting in Los Angeles, California. The meeting features two specific poster sessions for emerging scholars:

1. A poster session on Friday, April 10, 2026, showcasing dissertation fellows and their research
2. A poster session on Saturday, April 11, 2026, featuring early career scholars and their work

Provide the following information about these two poster sessions:

For Each Session:
- The official session name (as listed in the conference program)
- The exact date
- The session time (start time to end time, in Pacific Time)
- The specific location (building and room/hall name)

Additionally, identify one poster presentation from each session:
- From the Friday session: Identify one poster where the research focuses on student learning, student engagement, or academic outcomes
- From the Saturday session: Identify one poster where the research focuses on teacher education, professional development, or educator preparation

For each identified poster, provide:
- The presentation title
- The presenting author's full name
- The author's institutional affiliation
"""

EXPECTED = {
    "friday": {
        "session_name": "AERA Promising Scholarship in Education Research: Dissertation Fellows and Their Research",
        "date": "Friday, April 10, 2026",
        "time": "11:45 am to 1:15 pm Pacific Time",
        "location": "Los Angeles Convention Center, Poster Hall - Exhibit Hall A",
        "focus_categories": [
            "student learning",
            "student engagement",
            "academic outcomes"
        ],
    },
    "saturday": {
        "session_name": "Excellence in Education Research: Early Career Scholars and Their Work",
        "date": "Saturday, April 11, 2026",
        "time": "11:45 am to 1:15 pm Pacific Time",
        "location": "Los Angeles Convention Center, Poster Hall - Exhibit Hall A",
        "focus_categories": [
            "teacher education",
            "professional development",
            "educator preparation"
        ],
    },
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SessionInfo(BaseModel):
    session_name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    location: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PosterInfo(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    institution: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AERAExtraction(BaseModel):
    friday_session: Optional[SessionInfo] = None
    friday_poster: Optional[PosterInfo] = None
    saturday_session: Optional[SessionInfo] = None
    saturday_poster: Optional[PosterInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_aera() -> str:
    return """
    Extract structured information from the answer about two specific AERA 2026 poster sessions and one qualifying poster from each.

    For each of the following, extract exactly what the answer states (do not infer):
    1) friday_session:
       - session_name: official session name as written in the answer
       - date: exact date string as written (e.g., "Friday, April 10, 2026")
       - time: session time window with timezone as written (e.g., "11:45 am to 1:15 pm Pacific Time")
       - location: building and room/hall name as written (e.g., "Los Angeles Convention Center, Poster Hall - Exhibit Hall A")
       - sources: ALL URLs in the answer that support this session’s details (program pages, schedule pages, etc.)

    2) friday_poster:
       - title: the selected poster title from the Friday session
       - author: the presenting author's full name
       - institution: the author's institutional affiliation
       - sources: ALL URLs in the answer that support this poster (the session program page or a dedicated page for the poster)

    3) saturday_session: (same fields as friday_session)
       - session_name
       - date
       - time
       - location
       - sources

    4) saturday_poster: (same fields as friday_poster)
       - title
       - author
       - institution
       - sources

    Rules:
    - Return null for any field not present in the answer.
    - For sources, return only valid URLs explicitly present in the answer text. If none are present, return an empty list.
    - Do not invent or transform information. Preserve capitalization and punctuation from the answer.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _safe_sources(*lists: Optional[List[str]]) -> List[str]:
    """Combine and deduplicate multiple possible URL lists, skipping Nones."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if isinstance(url, str):
                u = url.strip()
                if u and u not in seen:
                    combined.append(u)
                    seen.add(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_session(
    evaluator: Evaluator,
    parent_node,
    day_key: str,
    session_data: Optional[SessionInfo],
) -> None:
    """
    Build leaves for a session (name, date, time, location) and verify each against sources and expected values.
    """
    expected = EXPECTED[day_key]
    day_readable = day_key.capitalize()

    # Build leaf nodes
    name_node = evaluator.add_leaf(
        id=f"{day_key}_session_name",
        desc=f"Provide the official name of the {day_readable} session as listed in the AERA 2026 program (must match expected)",
        parent=parent_node,
        critical=True,
    )
    date_node = evaluator.add_leaf(
        id=f"{day_key}_session_date",
        desc=f"Provide the exact date of the {day_readable} poster session (must match expected)",
        parent=parent_node,
        critical=True,
    )
    time_node = evaluator.add_leaf(
        id=f"{day_key}_session_time",
        desc=f"Provide the session time with start and end times in Pacific Time for the {day_readable} session (must match expected)",
        parent=parent_node,
        critical=True,
    )
    loc_node = evaluator.add_leaf(
        id=f"{day_key}_session_location",
        desc=f"Provide the specific location including building and room/hall name for the {day_readable} session (must match expected)",
        parent=parent_node,
        critical=True,
    )

    # Prepare claims
    sess_name = session_data.session_name if session_data else None
    sess_date = session_data.date if session_data else None
    sess_time = session_data.time if session_data else None
    sess_loc = session_data.location if session_data else None
    sess_sources = session_data.sources if session_data else []

    # Combine clear, targeted claims that both (a) tie to the answer and (b) check against the official expected string,
    # and request support from provided URLs.
    name_claim = (
        f"The session name reported in the answer is '{sess_name}'. "
        f"The official session name in the AERA 2026 program for the {day_readable} session is "
        f"'{expected['session_name']}'. These two should be the same (allow minor case/whitespace differences)."
    )
    date_claim = (
        f"The session date reported in the answer is '{sess_date}'. "
        f"The correct date in the AERA 2026 program for the {day_readable} session is '{expected['date']}'. "
        f"These two should be the same (allow minor formatting differences only)."
    )
    time_claim = (
        f"The session time reported in the answer is '{sess_time}'. "
        f"The correct time window for the {day_readable} session is '{expected['time']}'. "
        f"These two should be the same (allow minor formatting differences only)."
    )
    loc_claim = (
        f"The session location reported in the answer is '{sess_loc}'. "
        f"The correct location for the {day_readable} session is '{expected['location']}'. "
        f"These two should be the same (allow minor punctuation/spacing differences only)."
    )

    # Additional instructions per field
    add_ins_common = (
        "Use the provided conference program URLs to verify the information. "
        "Treat simple variations in capitalization or whitespace as equivalent, but the substance must match exactly."
    )

    # Verify
    await evaluator.verify(claim=name_claim, node=name_node, sources=sess_sources, additional_instruction=add_ins_common)
    await evaluator.verify(claim=date_claim, node=date_node, sources=sess_sources, additional_instruction=add_ins_common)
    await evaluator.verify(claim=time_claim, node=time_node, sources=sess_sources, additional_instruction=add_ins_common)
    await evaluator.verify(claim=loc_claim, node=loc_node, sources=sess_sources, additional_instruction=add_ins_common)


async def build_and_verify_poster(
    evaluator: Evaluator,
    parent_node,
    day_key: str,
    poster: Optional[PosterInfo],
    session: Optional[SessionInfo],
) -> None:
    """
    Add and verify three leaves for a selected poster from a session:
    - Title leaf (also implicitly validates topical focus requirement and session membership)
    - Presenting author name
    - Presenting author's institution
    """
    expected = EXPECTED[day_key]
    day_readable = day_key.capitalize()

    # Leaf nodes
    title_node = evaluator.add_leaf(
        id=f"{day_key}_poster_title",
        desc=f"Provide the title of one poster from the {day_readable} session that meets the required topical focus",
        parent=parent_node,
        critical=True,
    )
    author_node = evaluator.add_leaf(
        id=f"{day_key}_poster_author",
        desc=f"Provide the presenting author's full name for the identified {day_readable} poster",
        parent=parent_node,
        critical=True,
    )
    inst_node = evaluator.add_leaf(
        id=f"{day_key}_poster_institution",
        desc=f"Provide the institutional affiliation of the presenting author for the identified {day_readable} poster",
        parent=parent_node,
        critical=True,
    )

    # Data and sources
    title = poster.title if poster else None
    author = poster.author if poster else None
    inst = poster.institution if poster else None
    poster_sources = poster.sources if poster else []
    session_sources = session.sources if session else []
    all_sources = _safe_sources(poster_sources, session_sources)

    # Focus categories wording
    if day_key == "friday":
        focus_text = "student learning, student engagement, or academic outcomes"
    else:
        focus_text = "teacher education, professional development, or educator preparation"

    # Claims
    title_claim = (
        f"The identified {day_readable} poster is titled '{title}'. "
        f"It appears in the AERA 2026 {day_readable} session "
        f"'{expected['session_name']}', and the poster's topic focuses on {focus_text}. "
        f"Confirm via the provided program/source URLs."
    )
    author_claim = (
        f"The presenting author for the poster titled '{title}' is '{author}'. "
        f"Verify this using the provided program/source URLs."
    )
    inst_claim = (
        f"The institutional affiliation of the presenting author '{author}' for the poster titled '{title}' is '{inst}'. "
        f"Verify this using the provided program/source URLs."
    )

    # Additional instruction
    add_ins_title = (
        "Verify that: (1) the poster belongs to the specified session and AERA 2026; "
        f"(2) the topical focus clearly aligns with one or more of: {focus_text} "
        "(use abstract/keywords/area if available); "
        "(3) the provided title matches what is on the program page (allow minor case/whitespace differences)."
    )
    add_ins_simple = (
        "Rely on the official program/source URLs. Allow minor formatting/casing differences, "
        "but the name/affiliation must substantively match."
    )

    # Verify
    await evaluator.verify(claim=title_claim, node=title_node, sources=all_sources, additional_instruction=add_ins_title)
    await evaluator.verify(claim=author_claim, node=author_node, sources=all_sources, additional_instruction=add_ins_simple)
    await evaluator.verify(claim=inst_claim, node=inst_node, sources=all_sources, additional_instruction=add_ins_simple)


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
    Evaluate an answer for the AERA 2026 poster sessions task.
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

    # Record expected (ground truth) info for transparency
    evaluator.add_ground_truth({
        "expected_friday": EXPECTED["friday"],
        "expected_saturday": EXPECTED["saturday"],
        "notes": "We verify the provided answer against the official AERA 2026 program pages cited in the answer."
    })

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_aera(),
        template_class=AERAExtraction,
        extraction_name="aera_2026_sessions_and_posters",
    )

    # Build rubric subtrees according to the provided JSON structure

    # 1) Friday session info (parallel, critical: false)
    friday_info_node = evaluator.add_parallel(
        id="friday_session_info",
        desc="Provide all required information about the Friday dissertation fellows poster session",
        parent=root,
        critical=False,
    )
    await build_and_verify_session(
        evaluator=evaluator,
        parent_node=friday_info_node,
        day_key="friday",
        session_data=extracted.friday_session,
    )

    # 2) Friday poster (parallel, critical: false)
    friday_poster_node = evaluator.add_parallel(
        id="friday_poster",
        desc="Identify one qualifying poster from the Friday session and provide required details",
        parent=root,
        critical=False,
    )
    await build_and_verify_poster(
        evaluator=evaluator,
        parent_node=friday_poster_node,
        day_key="friday",
        poster=extracted.friday_poster,
        session=extracted.friday_session,
    )

    # 3) Saturday session info (parallel, critical: false)
    saturday_info_node = evaluator.add_parallel(
        id="saturday_session_info",
        desc="Provide all required information about the Saturday early career scholars poster session",
        parent=root,
        critical=False,
    )
    await build_and_verify_session(
        evaluator=evaluator,
        parent_node=saturday_info_node,
        day_key="saturday",
        session_data=extracted.saturday_session,
    )

    # 4) Saturday poster (parallel, critical: false)
    saturday_poster_node = evaluator.add_parallel(
        id="saturday_poster",
        desc="Identify one qualifying poster from the Saturday session and provide required details",
        parent=root,
        critical=False,
    )
    await build_and_verify_poster(
        evaluator=evaluator,
        parent_node=saturday_poster_node,
        day_key="saturday",
        poster=extracted.saturday_poster,
        session=extracted.saturday_session,
    )

    # Final summary
    return evaluator.get_summary()