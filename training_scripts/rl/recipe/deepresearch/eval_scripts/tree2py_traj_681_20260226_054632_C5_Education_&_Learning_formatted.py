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
TASK_ID = "sacscoc_fbs_eng_grad_selection"
TASK_DESCRIPTION = (
    "A prospective graduate student is researching universities for pursuing a Master's degree in engineering. "
    "The student has identified several important criteria for their university search:\n\n"
    "1. The university must be accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC)\n"
    "2. The university must have an NCAA Division I Football Bowl Subdivision (FBS) football program\n"
    "3. The university must offer graduate-level engineering degree programs (Master's and/or Doctoral programs)\n"
    "4. The university must be located in one of the following states: Tennessee, Georgia, or Florida\n\n"
    "Identify at least two universities that meet ALL of the above criteria. For each university you identify, provide the following information:\n\n"
    "- University name\n"
    "- Verification of SACSCOC accreditation status for graduate programs\n"
    "- Verification of NCAA Division I FBS football participation, including the specific athletic conference\n"
    "- Description of the graduate engineering programs offered\n"
    "- The city and state where the university's main campus is located\n"
    "- Reference URLs from official sources (university websites, SACSCOC website, conference websites, etc.) that support your findings\n\n"
    "Ensure all information is accurate and can be verified through the provided reference URLs."
)

ELIGIBLE_STATES = {"tennessee", "georgia", "florida"}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    location_url: Optional[str] = None

    sacscoc_url: Optional[str] = None

    fbs_url: Optional[str] = None
    fbs_conference: Optional[str] = None
    conference_url: Optional[str] = None

    engineering_desc: Optional[str] = None
    engineering_url: Optional[str] = None


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to the first three universities mentioned in the answer that are intended to meet the specified criteria.
    For each university, extract the following fields exactly as reported in the answer (use null if missing):

    - name: University name string.
    - city: The city of the university's main campus.
    - state: The state of the university's main campus (either full name like "Georgia" or common postal abbreviation like "GA").
    - location_url: A single official URL that supports the main campus city and state (prefer official university pages such as "About", "Contact", "Campuses", or "Directions"; if unavailable, use another official profile page with location).
    - sacscoc_url: A single official URL proving SACSCOC accreditation (either the SACSCOC institution listing page or the university's official accreditation page explicitly referencing SACSCOC).
    - fbs_url: A single official URL supporting NCAA Division I FBS football participation for the university (prefer official athletics site, NCAA/conference pages that clearly indicate FBS participation).
    - fbs_conference: The name of the university's FBS football conference (e.g., "SEC", "ACC", "Big 12", "American Athletic Conference", "Sun Belt", "Conference USA", etc.).
    - conference_url: A single official URL supporting membership in the reported FBS conference (prefer official conference or the university athletics page that lists conference affiliation).
    - engineering_desc: A brief description of the graduate-level engineering programs offered (Master's and/or Doctoral), taken verbatim or summarized from the answer.
    - engineering_url: A single official URL supporting the existence of graduate-level engineering degree programs (e.g., engineering college page, graduate catalog/programs page).

    Return an object with a 'universities' array containing up to three university objects with these fields.
    IMPORTANT:
    - Extract only URLs explicitly present in the answer. If the answer mentions a source but does not give a URL, set the URL field to null.
    - If a URL is missing a protocol (http/https), prepend "http://".
    - Do not invent any information; if a field is missing in the answer, set it to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _has_text(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _state_str_normalized(state: Optional[str]) -> str:
    if not _has_text(state):
        return ""
    s = state.strip().lower()
    # Map common postal abbreviations to full names
    mapping = {
        "tn": "tennessee",
        "ga": "georgia",
        "fl": "florida"
    }
    return mapping.get(s, s)


def _eligible_state(state: Optional[str]) -> bool:
    return _state_str_normalized(state) in ELIGIBLE_STATES


# --------------------------------------------------------------------------- #
# Verification for a single university                                        #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    index_one_based: int
) -> Any:
    """
    Build and verify the subtree for a single university with critical checks.
    """
    u_node = evaluator.add_parallel(
        id=f"University_{index_one_based}",
        desc=f"University entry #{index_one_based} (independent item).",
        parent=parent_node,
        critical=False
    )

    # Name provided (critical)
    evaluator.add_custom_node(
        result=_has_text(uni.name),
        id=f"U{index_one_based}_Name_Provided",
        desc="University name is provided.",
        parent=u_node,
        critical=True
    )

    # Location existence gate (critical custom, to enforce source-grounding)
    evaluator.add_custom_node(
        result=_has_text(uni.city) and _has_text(uni.state) and _has_text(uni.location_url),
        id=f"U{index_one_based}_Location_URL_Provided",
        desc="Main campus city/state and an official location URL are provided.",
        parent=u_node,
        critical=True
    )

    # Location verification (critical)
    loc_leaf = evaluator.add_leaf(
        id=f"U{index_one_based}_Main_Campus_City_State_Provided_With_Official_URL",
        desc="Main campus city and state are provided and supported by a specific verifiable official URL.",
        parent=u_node,
        critical=True
    )
    loc_claim = f"The main campus of {uni.name or 'the university'} is located in {uni.city or '[city missing]'}, {uni.state or '[state missing]'}."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=uni.location_url if _has_text(uni.location_url) else None,
        additional_instruction=(
            "Verify the campus location (city and state) on the provided official page. "
            "Allow minor formatting variants (e.g., 'GA' vs 'Georgia'). "
            "If the page lists multiple campuses, prefer the main campus or headquarters."
        )
    )

    # State eligibility (critical)
    state_leaf = evaluator.add_leaf(
        id=f"U{index_one_based}_State_Is_Eligible",
        desc="University is located in Tennessee, Georgia, or Florida.",
        parent=u_node,
        critical=True
    )
    state_norm = _state_str_normalized(uni.state)
    elig_claim = f"The state '{uni.state or '[state missing]'}' is one of Tennessee, Georgia, or Florida."
    # We still pass the location_url as supporting evidence for state correctness.
    await evaluator.verify(
        claim=elig_claim,
        node=state_leaf,
        sources=uni.location_url if _has_text(uni.location_url) else None,
        additional_instruction=(
            "Judge whether the provided state is among the allowed set. "
            "Use the official location page as evidence for the state. "
            "Treat 'GA' as Georgia, 'TN' as Tennessee, and 'FL' as Florida."
        )
    )

    # SACSCOC accreditation gate (critical)
    evaluator.add_custom_node(
        result=_has_text(uni.sacscoc_url),
        id=f"U{index_one_based}_SACSCOC_URL_Provided",
        desc="An official SACSCOC accreditation URL is provided.",
        parent=u_node,
        critical=True
    )

    # SACSCOC accreditation verification (critical)
    sacscoc_leaf = evaluator.add_leaf(
        id=f"U{index_one_based}_SACSCOC_Accreditation_For_Graduate_Degrees_With_Official_URL",
        desc="SACSCOC accreditation to award graduate degrees is stated and supported by a specific official URL (SACSCOC or official university accreditation page).",
        parent=u_node,
        critical=True
    )
    sac_claim = (
        f"{uni.name or 'The university'} is accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC) "
        f"to award graduate degrees (Master's and/or Doctoral)."
    )
    await evaluator.verify(
        claim=sac_claim,
        node=sacscoc_leaf,
        sources=uni.sacscoc_url,
        additional_instruction=(
            "Confirm that the page explicitly references SACSCOC accreditation and includes authority to award graduate degrees "
            "(e.g., 'master', 'graduate', 'doctoral'). Either a SACSCOC institution listing or an official university accreditation page is acceptable."
        )
    )

    # FBS participation gate (critical)
    evaluator.add_custom_node(
        result=_has_text(uni.fbs_url),
        id=f"U{index_one_based}_FBS_URL_Provided",
        desc="An official FBS participation URL is provided.",
        parent=u_node,
        critical=True
    )

    # FBS participation verification (critical)
    fbs_leaf = evaluator.add_leaf(
        id=f"U{index_one_based}_FBS_Football_With_Official_URL",
        desc="NCAA Division I FBS football participation is stated and supported by a specific official URL (official athletics/conference/NCAA-relevant official page).",
        parent=u_node,
        critical=True
    )
    fbs_claim = f"{uni.name or 'The university'} participates in NCAA Division I FBS football."
    await evaluator.verify(
        claim=fbs_claim,
        node=fbs_leaf,
        sources=uni.fbs_url,
        additional_instruction=(
            "Verify that the page supports that the university's football program is at the FBS level. "
            "Accept official athletics pages, official conference pages, or NCAA pages that clearly indicate FBS participation."
        )
    )

    # Conference gate (critical)
    evaluator.add_custom_node(
        result=_has_text(uni.fbs_conference) and _has_text(uni.conference_url),
        id=f"U{index_one_based}_Conference_URL_Provided",
        desc="The specific FBS conference name and an official URL are provided.",
        parent=u_node,
        critical=True
    )

    # Conference verification (critical)
    conf_leaf = evaluator.add_leaf(
        id=f"U{index_one_based}_FBS_Conference_Reported_With_Official_URL",
        desc="The specific FBS athletic conference is reported and supported by a specific official URL (official conference or official athletics page).",
        parent=u_node,
        critical=True
    )
    conf_name = uni.fbs_conference or "[conference missing]"
    conf_claim = f"{uni.name or 'The university'} is a member of the {conf_name} football conference at the FBS level."
    await evaluator.verify(
        claim=conf_claim,
        node=conf_leaf,
        sources=uni.conference_url,
        additional_instruction=(
            "Confirm that the page shows the university as a football member of the stated conference (e.g., SEC, ACC, Big 12, American, Sun Belt, Conference USA). "
            "Prefer official conference sites or official athletics pages listing conference affiliation."
        )
    )

    # Graduate engineering gate (critical)
    evaluator.add_custom_node(
        result=_has_text(uni.engineering_url),
        id=f"U{index_one_based}_Engineering_URL_Provided",
        desc="An official URL describing graduate-level engineering programs is provided.",
        parent=u_node,
        critical=True
    )

    # Graduate engineering verification (critical)
    eng_leaf = evaluator.add_leaf(
        id=f"U{index_one_based}_Graduate_Engineering_Programs_Described_With_Official_URL",
        desc="Graduate-level engineering programs (Master's and/or Doctoral) are described and supported by a specific official URL (engineering/graduate catalog page).",
        parent=u_node,
        critical=True
    )
    eng_claim = (
        f"{uni.name or 'The university'} offers graduate-level engineering degree programs (Master's and/or Doctoral)."
    )
    eng_ins = (
        "Verify on the provided page that graduate engineering programs exist "
        "(e.g., master's or PhD in engineering or engineering subfields). "
        "The following description from the answer is context, not a strict match requirement: "
        f"{uni.engineering_desc or '[no description provided]'}"
    )
    await evaluator.verify(
        claim=eng_claim,
        node=eng_leaf,
        sources=uni.engineering_url,
        additional_instruction=eng_ins
    )

    return u_node


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
    Evaluate an answer for the SACSCOC/FBS/Graduate Engineering selection task.
    """
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
        default_model=model
    )

    # Extract university entries from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Filter to first up to 3 universities for evaluation
    universities = (extracted.universities or [])[:3]
    # Pad if needed (to keep index logic stable)
    while len(universities) < 3:
        universities.append(UniversityItem())

    # Root child: At least two distinct universities listed (critical)
    names = [u.name for u in universities if _has_text(u.name)]
    distinct_names = set(_normalize_name(n) for n in names if _has_text(n))
    at_least_two_distinct = len([n for n in names if _has_text(n)]) >= 2 and len(distinct_names) >= 2

    evaluator.add_custom_node(
        result=at_least_two_distinct,
        id="At_Least_Two_Distinct_Universities_Listed",
        desc="Response lists at least two distinct universities by name.",
        parent=root,
        critical=True
    )

    # Per-university evaluations (non-critical parallel)
    per_unis_node = evaluator.add_parallel(
        id="Per_University_Evaluations",
        desc="Evaluate each provided university entry against the required constraints and required reported details (with official supporting URLs).",
        parent=root,
        critical=False
    )

    # Build and verify up to three university subtrees
    uni_nodes: List[Any] = []
    for i in range(3):
        uni = universities[i]
        node = await verify_university(evaluator, per_unis_node, uni, i + 1)
        uni_nodes.append(node)

    # Final critical check: at least two universities pass all per-university criteria
    # Compute each university node's aggregated score to determine pass/fail (1.0 == all critical checks passed)
    pass_count = 0
    for node in uni_nodes[:3]:
        try:
            score = node.compute_score(mutate=True)
        except Exception:
            score = 0.0
        if score == 1.0:
            pass_count += 1

    evaluator.add_custom_node(
        result=pass_count >= 2,
        id="At_Least_Two_Universities_Pass_All_Per_University_Criteria",
        desc="At least two of the evaluated university entries pass all their critical per-university checks (i.e., meet ALL constraints and provide all required verified details).",
        parent=root,
        critical=True
    )

    # Add custom info for transparency
    evaluator.add_custom_info(
        info={"eligible_states": sorted(list(ELIGIBLE_STATES)), "distinct_names_count": len(distinct_names), "pass_count": pass_count},
        info_type="evaluation_meta",
        info_name="meta_info"
    )

    return evaluator.get_summary()