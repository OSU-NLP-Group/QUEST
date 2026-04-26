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
TASK_ID = "ncaa_conf_champs_dec_2025"
TASK_DESCRIPTION = (
    "Identify three universities in the United States that competed in a major NCAA Division I conference "
    "championship game in December 2025. For each university, provide the following information: "
    "(1) The official name of the university and its location (state), "
    "(2) The university's NCAA Division I athletic conference affiliation, "
    "(3) Confirmation that the university competes in NCAA Division I athletics and the football subdivision (FBS or FCS), "
    "(4) The total student enrollment for Fall 2025 (must be at least 10,000 students), "
    "(5) Details about the conference championship game played in December 2025, including the date of the game, the opponent university, and the final score and outcome (win or loss), "
    "(6) Confirmation that the university is accredited by a regional accrediting agency, and "
    "(7) Whether the university is a public or private institution. "
    "For each piece of information provided, include a reference URL from an official or credible source that verifies the information."
)

ALLOWED_MAJOR_CONFERENCES = {
    # Short forms and full names for robustness
    "sec", "southeastern conference",
    "big ten", "big ten conference",
    "acc", "atlantic coast conference",
    "big 12", "big 12 conference",
    "pac-12", "pac 12", "pac-12 conference", "pac 12 conference",
    "ivy league",
    "big east", "big east conference",
}

RECOGNIZED_REGIONAL_ACCREDITORS_HINT = (
    "Examples of U.S. regional accreditors include: Higher Learning Commission (HLC), "
    "Middle States Commission on Higher Education (MSCHE), New England Commission of Higher Education (NECHE), "
    "Northwest Commission on Colleges and Universities (NWCCU), Southern Association of Colleges and Schools "
    "Commission on Colleges (SACSCOC), and WASC Senior College and University Commission (WSCUC)."
)

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    # Basic identity
    name: Optional[str] = None
    name_urls: List[str] = Field(default_factory=list)
    state: Optional[str] = None
    state_urls: List[str] = Field(default_factory=list)
    institution_type: Optional[str] = None  # "Public" or "Private"
    type_urls: List[str] = Field(default_factory=list)

    # Athletic conference
    conference_name: Optional[str] = None
    conference_urls: List[str] = Field(default_factory=list)

    # Division / football subdivision
    division: Optional[str] = None  # Should indicate NCAA Division I
    football_subdivision: Optional[str] = None  # "FBS" or "FCS"
    division_urls: List[str] = Field(default_factory=list)

    # Enrollment (Fall 2025)
    enrollment_total: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)

    # Conference championship game (December 2025)
    game_date: Optional[str] = None
    game_type: Optional[str] = None  # Should indicate "conference championship"
    opponent: Optional[str] = None
    final_score: Optional[str] = None
    outcome: Optional[str] = None  # "win" or "loss"
    game_urls: List[str] = Field(default_factory=list)

    # Accreditation
    accreditation_agency: Optional[str] = None
    accreditation_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
Extract up to THREE universities that the answer claims competed in a major NCAA Division I conference championship game in December 2025.
For each university, extract the following fields. If any field is missing from the answer, return null for the field (or [] for URL lists). Do NOT invent any information. Only extract URLs explicitly present in the answer.

Return a JSON object with a top-level key "universities" that is an array of up to 3 objects. Each object must have these fields:

- name: The university's official name (string; e.g., "University of Example")
- name_urls: URLs that verify the official name (array of strings; can be university's official 'About' or title page)
- state: The U.S. state in which the university is located (string; e.g., "Texas")
- state_urls: URLs that verify the location/state (array of strings)
- institution_type: Whether the university is public or private (string "Public" or "Private", case-insensitive)
- type_urls: URLs that verify the public/private status (array of strings)

- conference_name: The name of the athletic conference (string; e.g., "SEC", "Big Ten", "ACC", "Big 12", "Pac-12", "Ivy League", "Big East")
- conference_urls: URLs that verify the conference membership (array of strings)

- division: The NCAA division (should indicate "NCAA Division I") (string)
- football_subdivision: The football subdivision ("FBS" or "FCS") (string; case-insensitive)
- division_urls: URLs that verify NCAA Division I status and football subdivision (array of strings; NCAA or official athletics/conference pages are preferred)

- enrollment_total: The total student enrollment for Fall 2025 (string as presented in the answer; do not parse; e.g., "43,200")
- enrollment_urls: URLs that verify the Fall 2025 total enrollment and show at least 10,000 students (array of strings; prefer Common Data Set 2025-2026 or institutional fact books)

- game_date: The date of the conference championship game (string; e.g., "December 6, 2025" or "12/6/2025")
- game_type: Should indicate this was a conference championship game (string; e.g., "SEC Championship Game")
- opponent: The opponent university (string)
- final_score: The final score as presented (string; e.g., "28-24" or "24–21")
- outcome: "win" or "loss" from the perspective of the listed university (string; case-insensitive)
- game_urls: URLs that verify the game details above (array of strings; credible sources like official athletics site, conference site, NCAA, ESPN, etc.)

- accreditation_agency: The regional accrediting agency name (string; e.g., "Higher Learning Commission")
- accreditation_urls: URLs that verify accreditation by that regional agency (array of strings; e.g., accreditor or university accreditation page)

Important:
- Use [] for any URL fields when the answer does not provide URLs.
- If the answer lists more than three universities, extract the first three only (based on their order in the answer).
- Do not infer URLs. Extract only actual URLs that appear in the answer; these may be plain text or in markdown link format.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _norm_conf_name(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("pac 12", "pac-12")
    return s


def _is_major_conference(s: Optional[str]) -> bool:
    return _norm_conf_name(s) in ALLOWED_MAJOR_CONFERENCES


def _is_valid_outcome(outcome: Optional[str]) -> bool:
    return _norm(outcome) in {"win", "loss"}


def _is_valid_football_subdivision(sub: Optional[str]) -> bool:
    return _norm(sub) in {"fbs", "fcs"}


# --------------------------------------------------------------------------- #
# Verification functions for a single university                              #
# --------------------------------------------------------------------------- #
async def verify_university(evaluator: Evaluator, parent_node, uni: UniversityItem, idx: int) -> None:
    # University-level container
    univ_node = evaluator.add_parallel(
        id=f"University_{idx+1}",
        desc=f"University #{idx+1} verification",
        parent=parent_node,
        critical=False,  # allow partial credit across universities
    )

    prefix = f"U{idx+1}"

    # ----------------------------- Basic Info --------------------------------
    basic_node = evaluator.add_parallel(
        id=f"{prefix}_Basic_Info",
        desc=f"Basic identifying information for University {idx+1}",
        parent=univ_node,
        critical=True  # group is essential for the university
    )

    # Presence checks (critical siblings to gate verifications)
    evaluator.add_custom_node(
        result=bool(uni.name and uni.name.strip()),
        id=f"{prefix}_Name_present",
        desc=f"{prefix} official name is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(uni.name_urls) > 0,
        id=f"{prefix}_Name_ref_present",
        desc=f"{prefix} name reference URL(s) provided",
        parent=basic_node,
        critical=True
    )
    name_leaf = evaluator.add_leaf(
        id=f"{prefix}_Name",
        desc="Official name of the university provided",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official name of the university is '{uni.name or ''}'.",
        node=name_leaf,
        sources=uni.name_urls,
        additional_instruction="Verify the official institutional name from the provided URLs (prefer the university's official website). Allow minor stylistic variations (e.g., with/without 'The')."
    )

    evaluator.add_custom_node(
        result=bool(uni.state and uni.state.strip()),
        id=f"{prefix}_Location_present",
        desc=f"{prefix} state location is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(uni.state_urls) > 0,
        id=f"{prefix}_Location_ref_present",
        desc=f"{prefix} location reference URL(s) provided",
        parent=basic_node,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id=f"{prefix}_Location",
        desc="University is located in the United States",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni.name or 'The university'} is located in the U.S. state of {uni.state or ''}.",
        node=loc_leaf,
        sources=uni.state_urls,
        additional_instruction="Confirm that the university is in the specified U.S. state. Allow reasonable abbreviation variants (e.g., 'CA' for California)."
    )

    evaluator.add_custom_node(
        result=bool(uni.institution_type and uni.institution_type.strip()),
        id=f"{prefix}_Type_present",
        desc=f"{prefix} public/private status is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(uni.type_urls) > 0,
        id=f"{prefix}_Type_ref_present",
        desc=f"{prefix} type reference URL(s) provided",
        parent=basic_node,
        critical=True
    )
    type_leaf = evaluator.add_leaf(
        id=f"{prefix}_Type",
        desc="Public or private status is identified",
        parent=basic_node,
        critical=True
    )
    type_norm = _norm(uni.institution_type)
    type_phrase = "public" if type_norm == "public" else ("private" if type_norm == "private" else (uni.institution_type or ""))
    await evaluator.verify(
        claim=f"{uni.name or 'The university'} is a {type_phrase} institution.",
        node=type_leaf,
        sources=uni.type_urls,
        additional_instruction="Verify whether the university is public or private from official or credible sources (e.g., university or state system sites)."
    )

    # ------------------------- Athletic Conference ---------------------------
    conf_node = evaluator.add_parallel(
        id=f"{prefix}_Athletic_Conference",
        desc=f"Athletic conference affiliation information for University {idx+1}",
        parent=univ_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(uni.conference_name and uni.conference_name.strip()),
        id=f"{prefix}_Conference_name_present",
        desc=f"{prefix} conference name provided",
        parent=conf_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(uni.conference_urls) > 0,
        id=f"{prefix}_Conference_ref_present",
        desc=f"{prefix} conference membership reference URL(s) provided",
        parent=conf_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_major_conference(uni.conference_name),
        id=f"{prefix}_Conference_is_major",
        desc=f"{prefix} conference is one of the specified major NCAA Division I conferences",
        parent=conf_node,
        critical=True
    )
    conf_leaf = evaluator.add_leaf(
        id=f"{prefix}_Conference_Name",
        desc="University is a member of a major NCAA Division I conference (Big 12, Big East, Big Ten, SEC, ACC, Pac-12, or Ivy League)",
        parent=conf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni.name or 'The university'} is a member of the {uni.conference_name or ''}.",
        node=conf_leaf,
        sources=uni.conference_urls,
        additional_instruction="Verify conference membership from official conference or university athletics pages."
    )

    # ------------------------ Division / Subdivision -------------------------
    div_node = evaluator.add_parallel(
        id=f"{prefix}_Division_Status",
        desc=f"NCAA division and subdivision status for University {idx+1}",
        parent=univ_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(uni.division_urls) > 0,
        id=f"{prefix}_Division_ref_present",
        desc=f"{prefix} division/subdivision reference URL(s) provided",
        parent=div_node,
        critical=True
    )
    ncaa_div_leaf = evaluator.add_leaf(
        id=f"{prefix}_NCAA_Division",
        desc="University competes in NCAA Division I athletics",
        parent=div_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni.name or 'The university'} competes in NCAA Division I athletics.",
        node=ncaa_div_leaf,
        sources=uni.division_urls,
        additional_instruction="Confirm NCAA Division I status from NCAA, conference, or official athletics sources."
    )

    evaluator.add_custom_node(
        result=_is_valid_football_subdivision(uni.football_subdivision),
        id=f"{prefix}_Football_Subdivision_value_valid",
        desc=f"{prefix} football subdivision value is FBS or FCS",
        parent=div_node,
        critical=True
    )
    fb_sub_leaf = evaluator.add_leaf(
        id=f"{prefix}_Football_Subdivision",
        desc="Football program competes in FBS or FCS subdivision",
        parent=div_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni.name or 'The university'}'s football program competes in the {uni.football_subdivision or ''} subdivision.",
        node=fb_sub_leaf,
        sources=uni.division_urls,
        additional_instruction="Verify football subdivision (FBS or FCS) from NCAA, conference, or the school's athletics website."
    )

    # ----------------------------- Enrollment --------------------------------
    enr_node = evaluator.add_parallel(
        id=f"{prefix}_Enrollment",
        desc=f"Enrollment information for University {idx+1}",
        parent=univ_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(uni.enrollment_total and uni.enrollment_total.strip()),
        id=f"{prefix}_Enrollment_value_present",
        desc=f"{prefix} enrollment number provided",
        parent=enr_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(uni.enrollment_urls) > 0,
        id=f"{prefix}_Enrollment_ref_present",
        desc=f"{prefix} enrollment reference URL(s) provided",
        parent=enr_node,
        critical=True
    )
    enr_thresh_leaf = evaluator.add_leaf(
        id=f"{prefix}_Enrollment_Size",
        desc="Total enrollment is at least 10,000 students",
        parent=enr_node,
        critical=True
    )
    await evaluator.verify(
        claim="For Fall 2025, the total student enrollment was at least 10,000 students.",
        node=enr_thresh_leaf,
        sources=uni.enrollment_urls,
        additional_instruction="Confirm a Fall 2025 (or 2025–2026 academic year) total enrollment figure ≥10,000 from official institutional sources (e.g., Common Data Set, Fact Book, Institutional Research)."
    )

    # ------------------------ Conference Championship Game -------------------
    game_node = evaluator.add_parallel(
        id=f"{prefix}_Game_Information",
        desc=f"Conference championship game information for University {idx+1}",
        parent=univ_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(uni.game_urls) > 0,
        id=f"{prefix}_Game_ref_present",
        desc=f"{prefix} game details reference URL(s) provided",
        parent=game_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(uni.game_date and uni.game_date.strip()),
        id=f"{prefix}_Game_date_present",
        desc=f"{prefix} game date provided",
        parent=game_node,
        critical=True
    )
    game_date_leaf = evaluator.add_leaf(
        id=f"{prefix}_Game_Date",
        desc="Game was played in December 2025",
        parent=game_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The conference championship game took place on {uni.game_date or ''}, which is in December 2025.",
        node=game_date_leaf,
        sources=uni.game_urls,
        additional_instruction="Verify the game date and ensure it falls in December 2025."
    )

    evaluator.add_custom_node(
        result=bool(uni.game_type and uni.game_type.strip()),
        id=f"{prefix}_Game_type_present",
        desc=f"{prefix} game type provided",
        parent=game_node,
        critical=True
    )
    game_type_leaf = evaluator.add_leaf(
        id=f"{prefix}_Game_Type",
        desc="Game was a conference championship game",
        parent=game_node,
        critical=True
    )
    await evaluator.verify(
        claim="This game was a conference championship game.",
        node=game_type_leaf,
        sources=uni.game_urls,
        additional_instruction="Confirm that the event was a conference championship (e.g., 'SEC Championship Game', 'ACC Championship Game', etc.)."
    )

    evaluator.add_custom_node(
        result=bool(uni.opponent and uni.opponent.strip()),
        id=f"{prefix}_Opponent_present",
        desc=f"{prefix} opponent provided",
        parent=game_node,
        critical=True
    )
    opp_leaf = evaluator.add_leaf(
        id=f"{prefix}_Opponent",
        desc="Opponent university is identified",
        parent=game_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In that game, the opponent was {uni.opponent or ''}.",
        node=opp_leaf,
        sources=uni.game_urls,
        additional_instruction="Verify the named opponent is correct for that conference championship game."
    )

    evaluator.add_custom_node(
        result=bool(uni.final_score and uni.final_score.strip()) and _is_valid_outcome(uni.outcome),
        id=f"{prefix}_Result_present",
        desc=f"{prefix} final score and outcome provided",
        parent=game_node,
        critical=True
    )
    result_leaf = evaluator.add_leaf(
        id=f"{prefix}_Game_Result",
        desc="Final score and outcome (win/loss) provided",
        parent=game_node,
        critical=True
    )
    outcome_phrase = _norm(uni.outcome)
    outcome_text = "won" if outcome_phrase == "win" else ("lost" if outcome_phrase == "loss" else (uni.outcome or ""))
    await evaluator.verify(
        claim=f"The final score was {uni.final_score or ''}, and {uni.name or 'the university'} {outcome_text} the game.",
        node=result_leaf,
        sources=uni.game_urls,
        additional_instruction="Verify the final score and whether the listed university won or lost."
    )

    # ------------------------------ Accreditation ----------------------------
    accr_node = evaluator.add_parallel(
        id=f"{prefix}_Accreditation",
        desc=f"Accreditation status for University {idx+1}",
        parent=univ_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(uni.accreditation_agency and uni.accreditation_agency.strip()),
        id=f"{prefix}_Accreditation_agency_present",
        desc=f"{prefix} accreditation agency provided",
        parent=accr_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(uni.accreditation_urls) > 0,
        id=f"{prefix}_Accreditation_ref_present",
        desc=f"{prefix} accreditation reference URL(s) provided",
        parent=accr_node,
        critical=True
    )
    accred_leaf = evaluator.add_leaf(
        id=f"{prefix}_Regional_Accreditation",
        desc="University is accredited by a regional accrediting agency",
        parent=accr_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni.name or 'The university'} is accredited by {uni.accreditation_agency or ''}, a U.S. regional accrediting agency.",
        node=accred_leaf,
        sources=uni.accreditation_urls,
        additional_instruction="Confirm regional accreditation via the accreditor or university pages. "
                               + RECOGNIZED_REGIONAL_ACCREDITORS_HINT
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
    # Initialize evaluator (root parallel, non-critical to allow partial credit overall)
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
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Record custom info (allowed conferences)
    evaluator.add_custom_info(
        {"allowed_major_conferences": sorted(list(ALLOWED_MAJOR_CONFERENCES))},
        info_type="constraint_info",
        info_name="allowed_major_conferences"
    )

    # Step 1 Identification node (non-critical to maintain consistency; includes a critical distinctness check)
    step_node = evaluator.add_parallel(
        id="Step_1_Identification",
        desc="Identify three distinct universities that meet the specified criteria",
        parent=root,
        critical=False
    )

    # Prepare exactly 3 items (pad if fewer)
    universities = list(extracted.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityItem())

    # Distinctness check (critical child under non-critical parent is allowed)
    names_norm = [(_norm(u.name)) for u in universities if _norm(u.name)]
    distinct_ok = len(names_norm) == 3 and len(set(names_norm)) == 3
    evaluator.add_custom_node(
        result=distinct_ok,
        id="Distinct_Universities",
        desc="Three distinct university names identified (case-insensitive)",
        parent=step_node,
        critical=True
    )

    # Verify each university
    for i, uni in enumerate(universities):
        await verify_university(evaluator, step_node, uni, i)

    # Return summary
    return evaluator.get_summary()