import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nj_catholic_presidency_2024"
TASK_DESCRIPTION = (
    "An individual was appointed to a university presidency in the United States with an announcement made in April 2024. "
    "The appointment became effective on July 1, 2024. Prior to this appointment, the person served as Vice Provost of a specific area "
    "at the same institution. The person holds a doctoral degree and also holds an S.T.L. degree. The university where this person currently serves "
    "as president is located in New Jersey and has a Catholic affiliation. The person is also a Catholic priest. Identify this educational leader and "
    "provide the following information: (1) Full name, (2) Exact date of appointment announcement, (3) Exact effective start date of the presidency, "
    "(4) Full title of the previous administrative position held immediately before the current presidency, (5) Name of the institution where the previous "
    "position was held, (6) Types of doctoral-level degrees held, (7) Full official name of the current institution, (8) Current position title, and "
    "(9) Reference URLs verifying each piece of information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LeaderExtraction(BaseModel):
    # Identity
    full_name: Optional[str] = None
    name_sources: List[str] = Field(default_factory=list)

    # Dates
    appointment_announcement_date: Optional[str] = None
    announcement_sources: List[str] = Field(default_factory=list)

    effective_start_date: Optional[str] = None
    start_date_sources: List[str] = Field(default_factory=list)

    # Previous role
    previous_admin_role_title: Optional[str] = None
    prev_role_title_sources: List[str] = Field(default_factory=list)

    previous_role_institution: Optional[str] = None
    prev_role_institution_sources: List[str] = Field(default_factory=list)

    # Degrees
    doctoral_degree_types: List[str] = Field(default_factory=list)
    doctoral_sources: List[str] = Field(default_factory=list)

    has_stl_degree: Optional[bool] = None
    stl_sources: List[str] = Field(default_factory=list)

    # Current institution
    current_institution_official_name: Optional[str] = None
    current_institution_sources: List[str] = Field(default_factory=list)

    current_position_title: Optional[str] = None
    current_position_sources: List[str] = Field(default_factory=list)

    # Institution attributes
    institution_location: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)

    catholic_affiliation_sources: List[str] = Field(default_factory=list)

    # Person attribute
    is_catholic_priest: Optional[bool] = None
    priest_sources: List[str] = Field(default_factory=list)

    # Any extra URLs the answer included
    extra_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_leader_fields() -> str:
    return """
    Extract the requested information about the educational leader described in the answer. Return exactly the following fields.

    Required fields to extract:
    1) full_name: The individual's full name as given in the answer text.
    2) name_sources: A list of URL(s) cited in the answer that verify the person's full name.
    3) appointment_announcement_date: The exact public announcement date of the presidency (as stated in the answer).
    4) announcement_sources: A list of URL(s) cited in the answer that verify the announcement date.
    5) effective_start_date: The exact effective start date of the presidency (as stated in the answer).
    6) start_date_sources: A list of URL(s) cited in the answer that verify the effective start date.
    7) previous_admin_role_title: The full title of the immediately prior administrative role (should include 'Vice Provost' and the specific area).
    8) prev_role_title_sources: URL(s) cited in the answer that verify the previous role title.
    9) previous_role_institution: The name of the institution where the previous Vice Provost role was held.
    10) prev_role_institution_sources: URL(s) cited in the answer that verify the previous role institution.
    11) doctoral_degree_types: A list of doctoral-level degree type names (e.g., 'Ph.D.', 'S.T.D.', 'J.C.D.', 'Ed.D.', 'D.Min.', etc.) held by the person.
    12) doctoral_sources: URL(s) cited in the answer that verify the doctoral degree(s).
    13) has_stl_degree: A boolean indicating whether the person holds an S.T.L. (Licentiate in Sacred Theology) degree as stated in the answer.
    14) stl_sources: URL(s) cited in the answer that verify the S.T.L. degree.
    15) current_institution_official_name: The full official name of the current institution.
    16) current_institution_sources: URL(s) cited in the answer that verify the official name of the institution and/or the presidency.
    17) current_position_title: The current position title (must be President at the institution).
    18) current_position_sources: URL(s) cited in the answer that verify the current position.
    19) institution_location: The location of the current institution as stated in the answer (e.g., 'South Orange, New Jersey').
    20) location_sources: URL(s) cited in the answer that verify the location (New Jersey).
    21) catholic_affiliation_sources: URL(s) cited in the answer that verify the institution has a Catholic affiliation.
    22) is_catholic_priest: A boolean indicating whether the person is a Catholic priest, as stated in the answer.
    23) priest_sources: URL(s) cited in the answer that verify the person's Catholic priest status.
    24) extra_urls: Any additional URLs cited in the answer that do not clearly belong to the above categories.

    Rules:
    - Extract only what is explicitly provided in the answer.
    - For each '*_sources' field, extract only actual URLs mentioned in the answer. If none are present, return an empty list.
    - If any field is missing, set it to null (for single values) or [] (for lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in url_lists:
        for u in lst:
            if u and u not in seen:
                seen.add(u)
                result.append(u)
    return result


def _has_text(value: Optional[str]) -> bool:
    return bool(value and isinstance(value, str) and value.strip() != "")


def _contains_vice_provost(title: Optional[str]) -> bool:
    if not _has_text(title):
        return False
    return "vice provost" in title.lower()


def _contains_president(title: Optional[str]) -> bool:
    if not _has_text(title):
        return False
    return "president" in title.lower()


def _date_in_april_2024(date_str: Optional[str]) -> bool:
    if not _has_text(date_str):
        return False
    s = date_str.strip()
    s_lower = s.lower()

    patterns = [
        r"\bapril\s+\d{1,2},?\s*2024\b",
        r"\bapr\.?\s+\d{1,2},?\s*2024\b",
        r"\bapril\s+2024\b",
        r"\b2024[-/.]0?4[-/.]\d{1,2}\b",
        r"\b0?4[-/.]\d{1,2}[-/.]2024\b",
        r"\b04[-/]\d{1,2}[-/]2024\b",
        r"\b\d{1,2}[-/]0?4[-/]2024\b"
    ]
    if "april" in s_lower and "2024" in s_lower:
        return True
    for p in patterns:
        if re.search(p, s_lower):
            return True
    return False


def _date_is_july_1_2024(date_str: Optional[str]) -> bool:
    if not _has_text(date_str):
        return False
    s = date_str.strip()
    s_lower = s.lower()

    patterns = [
        r"\bjuly\s+1,?\s*2024\b",
        r"\bjul\.?\s+1,?\s*2024\b",
        r"\b2024[-/.]0?7[-/.]0?1\b",
        r"\b0?7[-/.]0?1[-/.]2024\b"
    ]
    # normalize punctuation for textual match
    s_norm = re.sub(r"[,\s]+", " ", s_lower)
    textual = "july 1 2024" in s_norm
    if textual:
        return True
    for p in patterns:
        if re.search(p, s_lower):
            return True
    return False


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_name(evaluator: Evaluator, parent, data: LeaderExtraction):
    node = evaluator.add_sequential(
        id="leader_full_name",
        desc="Provide the individual's full name and at least one reference URL that verifies the name.",
        parent=parent,
        critical=True,
    )
    exists = evaluator.add_custom_node(
        result=_has_text(data.full_name) and len(data.name_sources) > 0,
        id="leader_full_name_exists",
        desc="Full name is provided and at least one source URL is provided for the name.",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="leader_full_name_supported",
        desc="The provided full name is supported by the cited source(s).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The individual's full name is '{data.full_name or ''}'.",
        node=leaf,
        sources=data.name_sources,
        additional_instruction="Verify that the page(s) show the same full name, allowing minor formatting differences (e.g., titles like Rev., Msgr.).",
    )


async def verify_announcement_date(evaluator: Evaluator, parent, data: LeaderExtraction):
    node = evaluator.add_sequential(
        id="announcement_date",
        desc="Provide the exact public announcement date (must be in April 2024) and at least one reference URL verifying this date.",
        parent=parent,
        critical=True,
    )
    exists = evaluator.add_custom_node(
        result=_has_text(data.appointment_announcement_date) and len(data.announcement_sources) > 0,
        id="announcement_date_exists",
        desc="Announcement date and at least one source URL are provided.",
        parent=node,
        critical=True,
    )
    month_check = evaluator.add_custom_node(
        result=_date_in_april_2024(data.appointment_announcement_date),
        id="announcement_date_in_april_2024",
        desc="The provided announcement date is in April 2024.",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="announcement_date_supported",
        desc="The announcement date is supported by the cited source(s).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The announcement of the presidency was made on {data.appointment_announcement_date or ''}.",
        node=leaf,
        sources=data.announcement_sources,
        additional_instruction="Confirm the page explicitly states the announcement date; it should be in April 2024.",
    )


async def verify_effective_start_date(evaluator: Evaluator, parent, data: LeaderExtraction):
    node = evaluator.add_sequential(
        id="effective_start_date",
        desc="Provide the exact effective start date of the presidency (must be July 1, 2024) and at least one reference URL verifying this date.",
        parent=parent,
        critical=True,
    )
    exists = evaluator.add_custom_node(
        result=_has_text(data.effective_start_date) and len(data.start_date_sources) > 0,
        id="effective_start_date_exists",
        desc="Effective start date and at least one source URL are provided.",
        parent=node,
        critical=True,
    )
    july_check = evaluator.add_custom_node(
        result=_date_is_july_1_2024(data.effective_start_date),
        id="effective_start_date_is_july_1_2024",
        desc="The provided effective start date is July 1, 2024.",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="effective_start_date_supported",
        desc="The effective start date is supported by the cited source(s).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The presidency became effective on {data.effective_start_date or ''}.",
        node=leaf,
        sources=data.start_date_sources,
        additional_instruction="Confirm the page explicitly states the effective start date as July 1, 2024.",
    )


async def verify_previous_admin_role_title(evaluator: Evaluator, parent, data: LeaderExtraction):
    node = evaluator.add_sequential(
        id="previous_admin_role_title",
        desc="Provide the full title of the immediately previous administrative role (must be a Vice Provost role) and at least one reference URL verifying the title.",
        parent=parent,
        critical=True,
    )
    exists = evaluator.add_custom_node(
        result=_has_text(data.previous_admin_role_title) and len(data.prev_role_title_sources) > 0,
        id="previous_admin_role_title_exists",
        desc="Previous administrative role title and at least one source URL are provided.",
        parent=node,
        critical=True,
    )
    vp_check = evaluator.add_custom_node(
        result=_contains_vice_provost(data.previous_admin_role_title),
        id="previous_admin_role_is_vice_provost",
        desc="The previous role title includes 'Vice Provost' (case-insensitive).",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="previous_admin_role_title_supported",
        desc="The previous role title is supported by the cited source(s).",
        parent=node,
        critical=True,
    )
    combined_sources = _dedup_urls(data.prev_role_title_sources, data.prev_role_institution_sources)
    await evaluator.verify(
        claim=f"The person previously served as '{data.previous_admin_role_title or ''}'.",
        node=leaf,
        sources=combined_sources,
        additional_instruction="Verify that the page states the exact previous title, allowing minor formatting differences.",
    )


async def verify_previous_role_institution(evaluator: Evaluator, parent, data: LeaderExtraction):
    node = evaluator.add_sequential(
        id="previous_role_institution",
        desc="Provide the name of the institution where the previous Vice Provost role was held and at least one reference URL verifying it.",
        parent=parent,
        critical=True,
    )
    exists = evaluator.add_custom_node(
        result=_has_text(data.previous_role_institution) and len(data.prev_role_institution_sources) > 0,
        id="previous_role_institution_exists",
        desc="Previous role institution and at least one source URL are provided.",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="previous_role_institution_supported",
        desc="The previous role institution is supported by the cited source(s).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The previous Vice Provost role was held at '{data.previous_role_institution or ''}'.",
        node=leaf,
        sources=data.prev_role_institution_sources,
        additional_instruction="Verify that the page associates the Vice Provost role with the specified institution.",
    )


async def verify_same_institution_check(evaluator: Evaluator, parent, data: LeaderExtraction):
    node = evaluator.add_sequential(
        id="same_institution_check",
        desc="Confirm the previous Vice Provost role institution is the same as the current presidency institution.",
        parent=parent,
        critical=True,
    )
    exists = evaluator.add_custom_node(
        result=_has_text(data.previous_role_institution) and _has_text(data.current_institution_official_name),
        id="same_institution_fields_exist",
        desc="Both previous role institution and current institution official name are provided.",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="same_institution_match",
        desc="The previous role institution matches the current presidency institution.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The previous role institution '{data.previous_role_institution or ''}' and the current institution '{data.current_institution_official_name or ''}' refer to the same institution.",
        node=leaf,
        sources=None,
        additional_instruction="Allow common variants or abbreviations (e.g., 'Seton Hall' vs 'Seton Hall University'); judge if they are the same institution.",
    )


async def verify_stl_degree(evaluator: Evaluator, parent, data: LeaderExtraction):
    node = evaluator.add_sequential(
        id="stl_degree",
        desc="Confirm the person holds an S.T.L. degree and provide at least one reference URL verifying it.",
        parent=parent,
        critical=True,
    )
    exists = evaluator.add_custom_node(
        result=(len(data.stl_sources) > 0) and (data.has_stl_degree is True or True),
        id="stl_degree_sources_exist",
        desc="At least one source URL for the S.T.L. degree is provided.",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="stl_degree_supported",
        desc="The person holds an S.T.L. (Licentiate in Sacred Theology) degree as supported by the source(s).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The person holds an S.T.L. (Licentiate in Sacred Theology) degree.",
        node=leaf,
        sources=data.stl_sources,
        additional_instruction="Accept minor formatting differences (e.g., 'STL' vs 'S.T.L.'). Confirm the page explicitly states this degree.",
    )


async def verify_doctoral_degrees(evaluator: Evaluator, parent, data: LeaderExtraction):
    node = evaluator.add_sequential(
        id="doctoral_degree_types",
        desc="List the doctoral-level degree type(s) held and provide at least one reference URL verifying them.",
        parent=parent,
        critical=True,
    )
    exists = evaluator.add_custom_node(
        result=(len(data.doctoral_degree_types) > 0) and (len(data.doctoral_sources) > 0),
        id="doctoral_degrees_exist",
        desc="At least one doctoral degree type and at least one source URL are provided.",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="doctoral_degrees_supported",
        desc="The doctoral degree type(s) are supported by the cited source(s).",
        parent=node,
        critical=True,
    )
    listed = ", ".join(data.doctoral_degree_types) if data.doctoral_degree_types else ""
    await evaluator.verify(
        claim=f"The person holds the following doctoral-level degree(s): {listed}.",
        node=leaf,
        sources=data.doctoral_sources,
        additional_instruction="Allow reasonable variants of degree names (e.g., 'PhD' vs 'Ph.D.'). Confirm at least one doctoral degree is explicitly stated.",
    )


async def verify_current_institution_name(evaluator: Evaluator, parent, data: LeaderExtraction):
    node = evaluator.add_sequential(
        id="current_institution_name",
        desc="Provide the full official name of the current institution and at least one reference URL verifying it.",
        parent=parent,
        critical=True,
    )
    exists = evaluator.add_custom_node(
        result=_has_text(data.current_institution_official_name) and len(data.current_institution_sources) > 0,
        id="current_institution_name_exists",
        desc="Official name and at least one source URL are provided.",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="current_institution_name_supported",
        desc="The official name of the current institution is supported by the cited source(s).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The full official name of the current institution is '{data.current_institution_official_name or ''}'.",
        node=leaf,
        sources=data.current_institution_sources,
        additional_instruction="Confirm that the institution name on the page matches or is the official form.",
    )


async def verify_current_institution_nj(evaluator: Evaluator, parent, data: LeaderExtraction):
    node = evaluator.add_sequential(
        id="current_institution_nj",
        desc="Confirm the current institution is located in New Jersey.",
        parent=parent,
        critical=True,
    )
    sources = _dedup_urls(data.location_sources, data.current_institution_sources)
    exists = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="current_institution_nj_sources_exist",
        desc="At least one source URL for the New Jersey location is provided.",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="current_institution_nj_supported",
        desc="The current institution is located in New Jersey as supported by the source(s).",
        parent=node,
        critical=True,
    )
    inst_name = data.current_institution_official_name or "the institution"
    await evaluator.verify(
        claim=f"The institution '{inst_name}' is located in New Jersey.",
        node=leaf,
        sources=sources,
        additional_instruction="Verify that the page makes clear the institution is in New Jersey, USA. Minor variants like 'NJ' or city names are fine.",
    )


async def verify_current_institution_catholic(evaluator: Evaluator, parent, data: LeaderExtraction):
    node = evaluator.add_sequential(
        id="current_institution_catholic",
        desc="Confirm the current institution has a Catholic affiliation.",
        parent=parent,
        critical=True,
    )
    sources = _dedup_urls(data.catholic_affiliation_sources, data.current_institution_sources)
    exists = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="current_institution_catholic_sources_exist",
        desc="At least one source URL for the Catholic affiliation is provided.",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="current_institution_catholic_supported",
        desc="The institution's Catholic affiliation is supported by the source(s).",
        parent=node,
        critical=True,
    )
    inst_name = data.current_institution_official_name or "the institution"
    await evaluator.verify(
        claim=f"The institution '{inst_name}' has a Catholic affiliation.",
        node=leaf,
        sources=sources,
        additional_instruction="Verify that the page explicitly indicates Catholic identity, mission, sponsorship, or affiliation.",
    )


async def verify_current_position_title(evaluator: Evaluator, parent, data: LeaderExtraction):
    node = evaluator.add_sequential(
        id="current_position_title",
        desc="Provide the current position title (must be President at the institution) and at least one reference URL verifying it.",
        parent=parent,
        critical=True,
    )
    exists = evaluator.add_custom_node(
        result=_has_text(data.current_position_title) and len(data.current_position_sources) > 0,
        id="current_position_title_exists",
        desc="Current position title and at least one source URL are provided.",
        parent=node,
        critical=True,
    )
    pres_check = evaluator.add_custom_node(
        result=_contains_president(data.current_position_title),
        id="current_position_title_contains_president",
        desc="The current position title includes 'President' (case-insensitive).",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="current_position_title_supported",
        desc="The current position title is supported by the cited source(s).",
        parent=node,
        critical=True,
    )
    inst_name = data.current_institution_official_name or ""
    await evaluator.verify(
        claim=f"The person currently serves as {data.current_position_title or ''} at {inst_name}.",
        node=leaf,
        sources=data.current_position_sources,
        additional_instruction="Confirm the current position title; allow minor formatting differences (e.g., with or without institution name).",
    )


async def verify_priest_status(evaluator: Evaluator, parent, data: LeaderExtraction):
    node = evaluator.add_sequential(
        id="catholic_priest_status",
        desc="Confirm the person is a Catholic priest and provide at least one reference URL verifying this status.",
        parent=parent,
        critical=True,
    )
    exists = evaluator.add_custom_node(
        result=(data.is_catholic_priest is True) and (len(data.priest_sources) > 0),
        id="priest_status_sources_exist",
        desc="Catholic priest status is indicated and at least one source URL is provided.",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="priest_status_supported",
        desc="The person is a Catholic priest as supported by the source(s).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The individual is a Catholic priest.",
        node=leaf,
        sources=data.priest_sources,
        additional_instruction="Confirm the page (e.g., bio or announcement) explicitly identifies the person as a Catholic priest.",
    )


async def verify_source_reliability(evaluator: Evaluator, parent, data: LeaderExtraction):
    # Build the comprehensive set of URLs
    all_urls = _dedup_urls(
        data.name_sources,
        data.announcement_sources,
        data.start_date_sources,
        data.prev_role_title_sources,
        data.prev_role_institution_sources,
        data.doctoral_sources,
        data.stl_sources,
        data.current_institution_sources,
        data.current_position_sources,
        data.location_sources,
        data.catholic_affiliation_sources,
        data.priest_sources,
        data.extra_urls,
    )

    main = evaluator.add_parallel(
        id="source_reliability_main",
        desc="All provided references are from official institutional sources or reliable news sources, as required.",
        parent=parent,
        critical=True,
    )

    # If no URLs at all, add a failing custom node
    if not all_urls:
        evaluator.add_custom_node(
            result=False,
            id="source_reliability_any_urls",
            desc="At least one reference URL is provided overall.",
            parent=main,
            critical=True,
        )
        return

    # For each URL, verify reliability individually (all critical)
    for i, url in enumerate(all_urls):
        leaf = evaluator.add_leaf(
            id=f"source_reliability_url_{i+1}",
            desc=f"URL {i+1} is from an official institutional domain or a reliable news outlet.",
            parent=main,
            critical=True,
        )
        await evaluator.verify(
            claim="This webpage is an official institutional page or a reliable news outlet.",
            node=leaf,
            sources=url,
            additional_instruction=(
                "Judge reliability by the domain and on-page indicators. "
                "Accept: official university or seminary sites (often .edu or clearly branded), diocesan/archdiocesan sites, "
                "and reputable news outlets (national or well-known local/regional)."
            ),
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
    Evaluate the answer for the NJ Catholic presidency 2024 task and return the evaluation summary.
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

    # Create a critical root aggregator node to mirror rubric's critical root
    top = evaluator.add_parallel(
        id="Root",
        desc="Identify the educational leader meeting the stated conditions in the proposed question and provide all requested fields with verifiable citations.",
        parent=root,
        critical=True,
    )

    # Extract structured information
    extracted: LeaderExtraction = await evaluator.extract(
        prompt=prompt_extract_leader_fields(),
        template_class=LeaderExtraction,
        extraction_name="leader_extraction",
    )

    # Add some custom info for debugging
    evaluator.add_custom_info(
        info={
            "extracted_name": extracted.full_name,
            "extracted_current_institution": extracted.current_institution_official_name,
            "extracted_prev_role_title": extracted.previous_admin_role_title,
            "num_total_urls": len(_dedup_urls(
                extracted.name_sources,
                extracted.announcement_sources,
                extracted.start_date_sources,
                extracted.prev_role_title_sources,
                extracted.prev_role_institution_sources,
                extracted.doctoral_sources,
                extracted.stl_sources,
                extracted.current_institution_sources,
                extracted.current_position_sources,
                extracted.location_sources,
                extracted.catholic_affiliation_sources,
                extracted.priest_sources,
                extracted.extra_urls,
            )),
        },
        info_type="extraction_overview",
    )

    # Build and run verifications
    await verify_name(evaluator, top, extracted)
    await verify_announcement_date(evaluator, top, extracted)
    await verify_effective_start_date(evaluator, top, extracted)
    await verify_previous_admin_role_title(evaluator, top, extracted)
    await verify_previous_role_institution(evaluator, top, extracted)
    await verify_same_institution_check(evaluator, top, extracted)
    await verify_stl_degree(evaluator, top, extracted)
    await verify_doctoral_degrees(evaluator, top, extracted)
    await verify_current_institution_name(evaluator, top, extracted)
    await verify_current_institution_nj(evaluator, top, extracted)
    await verify_current_institution_catholic(evaluator, top, extracted)
    await verify_current_position_title(evaluator, top, extracted)
    await verify_priest_status(evaluator, top, extracted)
    await verify_source_reliability(evaluator, top, extracted)

    return evaluator.get_summary()