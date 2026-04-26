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
TASK_ID = "wh_communications_2025"
TASK_DESCRIPTION = """Research the current (2025) White House communications structure by identifying key officials and press corps members. Specifically, provide the following information:

1. White House Press Secretary: Identify the current White House Press Secretary who took office in January 2025. Provide their full name, the exact date they assumed the role, their age at appointment, and explain what historical record they hold related to their age. Also mention who previously held this record.

2. State Department Spokesperson: Identify the current State Department Spokesperson who was appointed in January 2025. Provide their full name, appointment date, and information about their professional background before joining the State Department.

3. White House Press Corps: Identify one current White House correspondent or reporter from each of the following three major news organizations:
   - The New York Times: Provide the reporter's name, official title, confirmation that they cover the White House specifically, and information about how long they have worked for the organization or their previous positions.
   - CNN: Provide the reporter's name, official title, confirmation of their White House beat, and details about when they joined CNN or their previous career positions.
   - Politico: Provide the reporter's name, official title, confirmation of their White House coverage, and information about their specific coverage focus or professional background.

For each of the five individuals identified (2 officials + 3 reporters), provide at least one authoritative reference URL from an official source (government website, news organization's official staff page, or professional profile).
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PressSecretaryInfo(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    start_date: Optional[str] = None
    age_at_appointment: Optional[str] = None
    historical_record: Optional[str] = None
    previous_record_holder_name: Optional[str] = None
    previous_record_holder_age: Optional[str] = None
    previous_record_holder_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StateSpoxInfo(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    start_date: Optional[str] = None
    previous_career: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ReporterInfo(BaseModel):
    org: Optional[str] = None
    name: Optional[str] = None
    title: Optional[str] = None
    white_house_role: Optional[str] = None
    background: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PressCorps(BaseModel):
    nyt: Optional[ReporterInfo] = None
    cnn: Optional[ReporterInfo] = None
    politico: Optional[ReporterInfo] = None


class CommsExtraction(BaseModel):
    press_secretary: Optional[PressSecretaryInfo] = None
    state_spox: Optional[StateSpoxInfo] = None
    reporters: Optional[PressCorps] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_comms() -> str:
    return """
    Extract structured information exactly as presented in the answer text for the following five individuals: the current White House Press Secretary (appointed in January 2025), the current State Department Spokesperson (appointed in January 2025), and one current White House correspondent/reporter each from The New York Times, CNN, and Politico.

    For each person, extract the fields listed below. Only extract values explicitly mentioned in the answer; do not invent information. If a field is missing, return null for that field. For URL fields, return every URL mentioned for that person, including official government or organization pages or official staff profiles; if no URLs are provided, return an empty array.

    Fields to extract:
    - press_secretary:
        - name
        - title (e.g., "White House Press Secretary")
        - start_date (exact date they took office, if provided)
        - age_at_appointment (the age at the time they assumed the role, if provided; leave as a string as written)
        - historical_record (a brief statement of the record they hold related to age, e.g., "youngest White House Press Secretary in U.S. history")
        - previous_record_holder_name (name of the previous record holder)
        - previous_record_holder_age (age of the previous record holder as written, e.g., "29")
        - previous_record_holder_year (year associated with that record, if provided, e.g., "1969")
        - sources (array of URLs for authoritative references)
    - state_spox:
        - name
        - title (e.g., "State Department Spokesperson")
        - start_date (appointment date)
        - previous_career (professional background before joining State; leave as a single string summarizing what the answer said)
        - sources (array of URLs)
    - reporters:
        - nyt (one person from The New York Times):
            - org (should be "The New York Times" or "NYT" if present; otherwise null)
            - name
            - title (official title)
            - white_house_role (text in the answer that confirms they cover the White House)
            - background (tenure at NYT or previous roles; leave as a single string)
            - sources (array of URLs)
        - cnn (one person from CNN):
            - org (should be "CNN" if present; otherwise null)
            - name
            - title (official title)
            - white_house_role (text confirming White House beat)
            - background (join date to CNN or previous positions; leave as a single string)
            - sources (array of URLs)
        - politico (one person from Politico):
            - org (should be "Politico" if present; otherwise null)
            - name
            - title (official title)
            - white_house_role (text confirming White House coverage)
            - background (coverage focus or professional background; leave as a single string)
            - sources (array of URLs)

    Notes:
    - Preserve text exactly as in the answer (including abbreviations). Dates should remain strings.
    - Only extract URLs that are explicitly present in the answer text (including markdown links). If a URL is missing a protocol, prepend http://
    - If multiple individuals per organization are listed, extract only the first one mentioned for that organization.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(s: Optional[str]) -> str:
    return s or ""


def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls or []


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_press_secretary(
    evaluator: Evaluator,
    parent_node,
    ps: Optional[PressSecretaryInfo],
) -> None:
    ps = ps or PressSecretaryInfo()
    ps_main = evaluator.add_parallel(
        id="ps_main",
        desc="Research the current White House Press Secretary",
        parent=parent_node,
        critical=True
    )

    # Reference must be evaluated first to gate other critical checks via auto preconditions
    ps_ref_leaf = evaluator.add_leaf(
        id="ps_reference",
        desc="Provide at least one authoritative reference URL for the White House Press Secretary information",
        parent=ps_main,
        critical=True
    )
    ps_name = _safe_str(ps.name)
    ps_title = _safe_str(ps.title)
    await evaluator.verify(
        claim=(
            f"This page is an authoritative source (e.g., whitehouse.gov, official news organization profile) "
            f"that confirms that {ps_name} serves as the {ps_title} appointed in January 2025."
        ),
        node=ps_ref_leaf,
        sources=_safe_urls(ps.sources),
        additional_instruction=(
            "Authoritative sources include: official government websites (e.g., whitehouse.gov), "
            "the individual's official bio on a major news organization's site, or a professional profile page. "
            "The page should clearly state the role and relate to the January 2025 appointment."
        )
    )

    # Basic information group
    ps_basic = evaluator.add_parallel(
        id="ps_basic",
        desc="Verify the identity, role, and appointment details of the current White House Press Secretary",
        parent=ps_main,
        critical=True
    )
    # Name & Title
    ps_name_title_leaf = evaluator.add_leaf(
        id="ps_name_title",
        desc="Provide the full name and official title of the current White House Press Secretary",
        parent=ps_basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ps_name} is the current White House Press Secretary of the United States.",
        node=ps_name_title_leaf,
        sources=_safe_urls(ps.sources),
        additional_instruction="Allow minor title variants like 'Press Secretary' vs 'White House Press Secretary'."
    )

    # Start date
    ps_start_date_leaf = evaluator.add_leaf(
        id="ps_start_date",
        desc="Provide the date when the current Press Secretary took office (January 2025)",
        parent=ps_basic,
        critical=True
    )
    ps_start_date = _safe_str(ps.start_date)
    await evaluator.verify(
        claim=f"{ps_name} took office as White House Press Secretary on {ps_start_date}.",
        node=ps_start_date_leaf,
        sources=_safe_urls(ps.sources),
        additional_instruction="Confirm the appointment date and that it is in January 2025. Accept reasonable date formatting variants."
    )

    # Age at appointment
    ps_age_leaf = evaluator.add_leaf(
        id="ps_age_at_appointment",
        desc="Provide the age of the Press Secretary at the time of appointment",
        parent=ps_basic,
        critical=True
    )
    ps_age = _safe_str(ps.age_at_appointment)
    await evaluator.verify(
        claim=f"At the time of appointment in January 2025, {ps_name} was {ps_age} years old.",
        node=ps_age_leaf,
        sources=_safe_urls(ps.sources),
        additional_instruction="Allow minor rounding differences (e.g., 29 vs 29.5)."
    )

    # Historical record group
    ps_record = evaluator.add_parallel(
        id="ps_record",
        desc="Verify and explain the historical record held by the current Press Secretary",
        parent=ps_main,
        critical=True
    )

    ps_youngest_leaf = evaluator.add_leaf(
        id="ps_youngest_record",
        desc="Verify and explain that the Press Secretary is the youngest in U.S. history to hold the position",
        parent=ps_record,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ps_name} is the youngest person in U.S. history to serve as White House Press Secretary.",
        node=ps_youngest_leaf,
        sources=_safe_urls(ps.sources),
        additional_instruction="The page should explicitly state or clearly imply 'youngest' record."
    )

    ps_prev_holder_leaf = evaluator.add_leaf(
        id="ps_previous_record_holder",
        desc="Identify who previously held the record as youngest Press Secretary (name, age, and year if provided)",
        parent=ps_record,
        critical=True
    )
    prev_name = _safe_str(ps.previous_record_holder_name)
    prev_age = _safe_str(ps.previous_record_holder_age)
    prev_year = _safe_str(ps.previous_record_holder_year)
    if prev_year:
        prev_claim = f"The previous youngest White House Press Secretary was {prev_name}, age {prev_age}, in {prev_year}."
    else:
        prev_claim = f"The previous youngest White House Press Secretary was {prev_name}, age {prev_age}."
    await evaluator.verify(
        claim=prev_claim,
        node=ps_prev_holder_leaf,
        sources=_safe_urls(ps.sources),
        additional_instruction="Check that the source mentions the previous record holder by name and age (and year if present)."
    )


async def verify_state_spokesperson(
    evaluator: Evaluator,
    parent_node,
    spox: Optional[StateSpoxInfo],
) -> None:
    spox = spox or StateSpoxInfo()
    state_main = evaluator.add_parallel(
        id="state_main",
        desc="Research the current State Department Spokesperson",
        parent=parent_node,
        critical=True
    )

    # Reference first (gating)
    state_ref_leaf = evaluator.add_leaf(
        id="state_reference",
        desc="Provide at least one authoritative reference URL for the State Department Spokesperson information",
        parent=state_main,
        critical=True
    )
    s_name = _safe_str(spox.name)
    s_title = _safe_str(spox.title)
    await evaluator.verify(
        claim=(
            f"This page is an authoritative source (e.g., state.gov or official profile) that confirms that "
            f"{s_name} serves as the {s_title} appointed in January 2025."
        ),
        node=state_ref_leaf,
        sources=_safe_urls(spox.sources),
        additional_instruction="Prefer state.gov pages or official organization profiles that clearly state the role and appointment timing."
    )

    # Basic information group
    state_basic = evaluator.add_parallel(
        id="state_basic",
        desc="Verify the identity, role, and appointment details of the current State Department Spokesperson",
        parent=state_main,
        critical=True
    )

    state_name_title_leaf = evaluator.add_leaf(
        id="state_name_title",
        desc="Provide the full name and official title of the current State Department Spokesperson",
        parent=state_basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"{s_name} is the current U.S. Department of State Spokesperson.",
        node=state_name_title_leaf,
        sources=_safe_urls(spox.sources),
        additional_instruction="Allow reasonable variants like 'State Department Spokesperson' vs 'Spokesperson for the Department of State'."
    )

    state_start_date_leaf = evaluator.add_leaf(
        id="state_start_date",
        desc="Provide the date when the current Spokesperson took office (January 2025)",
        parent=state_basic,
        critical=True
    )
    s_start_date = _safe_str(spox.start_date)
    await evaluator.verify(
        claim=f"{s_name} was appointed as State Department Spokesperson on {s_start_date}.",
        node=state_start_date_leaf,
        sources=_safe_urls(spox.sources),
        additional_instruction="Confirm that the appointment occurred in January 2025. Accept reasonable date formatting variants."
    )

    state_prev_career_leaf = evaluator.add_leaf(
        id="state_previous_career",
        desc="Provide information about the Spokesperson's professional background before joining the State Department",
        parent=state_basic,
        critical=True
    )
    s_prev = _safe_str(spox.previous_career)
    await evaluator.verify(
        claim=f"Before becoming State Department Spokesperson, {s_name}'s professional background included: {s_prev}",
        node=state_prev_career_leaf,
        sources=_safe_urls(spox.sources),
        additional_instruction="The page should mention prior roles, employers, beats, or relevant experience. Allow summary phrasing."
    )


async def verify_reporter(
    evaluator: Evaluator,
    parent_node,
    rep: Optional[ReporterInfo],
    org_key: str,
) -> None:
    # org_key expected values: "nyt", "cnn", "politico"
    rep = rep or ReporterInfo()
    org_display = {
        "nyt": "The New York Times",
        "cnn": "CNN",
        "politico": "Politico"
    }.get(org_key, _safe_str(rep.org) or org_key)

    base_id = f"{org_key}_reporter"
    main = evaluator.add_parallel(
        id=f"{base_id}_main",
        desc=f"Identify at least one current White House correspondent/reporter from {org_display}",
        parent=parent_node,
        critical=True
    )

    # Reference first
    ref_leaf = evaluator.add_leaf(
        id=f"{base_id}_reference",
        desc=f"Provide at least one authoritative reference URL ({org_display} official profile or bio)",
        parent=main,
        critical=True
    )
    r_name = _safe_str(rep.name)
    r_title = _safe_str(rep.title)
    await evaluator.verify(
        claim=(
            f"This page is an official {org_display} profile/bio or staff page that confirms "
            f"{r_name}'s role and White House coverage."
        ),
        node=ref_leaf,
        sources=_safe_urls(rep.sources),
        additional_instruction=(
            f"Prefer official {org_display} domains and staff profiles or author pages. "
            "The page should clearly state title and indicate White House coverage."
        )
    )

    # Details group
    details = evaluator.add_parallel(
        id=f"{base_id}_details",
        desc=f"Verify the identity, role, and professional background of the {org_display} White House reporter",
        parent=main,
        critical=True
    )

    name_title_leaf = evaluator.add_leaf(
        id=f"{org_key}_name_title",
        desc=f"Provide the full name and official title indicating White House correspondent/reporter role at {org_display}",
        parent=details,
        critical=True
    )
    await evaluator.verify(
        claim=f"{r_name} holds the title '{r_title}' at {org_display}.",
        node=name_title_leaf,
        sources=_safe_urls(rep.sources),
        additional_instruction="Allow minor variations in title wording. Accept equivalent titles that clearly indicate their position."
    )

    role_confirm_leaf = evaluator.add_leaf(
        id=f"{org_key}_white_house_role",
        desc="Confirm that the reporter currently covers the White House",
        parent=details,
        critical=True
    )
    r_role = _safe_str(rep.white_house_role)
    await evaluator.verify(
        claim=f"{r_name} currently covers the White House for {org_display}.",
        node=role_confirm_leaf,
        sources=_safe_urls(rep.sources),
        additional_instruction=(
            "The page should explicitly say 'White House' or clearly indicate the White House beat. "
            f"Supporting detail from the answer: {r_role}"
        )
    )

    background_leaf = evaluator.add_leaf(
        id=f"{org_key}_background",
        desc=f"Provide information about tenure or previous positions/background for {org_display} reporter",
        parent=details,
        critical=True
    )
    r_bg = _safe_str(rep.background)
    await evaluator.verify(
        claim=(
            f"This page includes information about {r_name}'s tenure at {org_display} or previous positions. "
            f"Specifically: {r_bg}"
        ),
        node=background_leaf,
        sources=_safe_urls(rep.sources),
        additional_instruction="The page should contain either tenure (when joined) or prior roles; allow summary phrasing."
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
    Evaluate an answer for the 2025 White House communications & press corps task.
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_comms(),
        template_class=CommsExtraction,
        extraction_name="comms_extraction"
    )

    # Build rubric tree (with a critical task root under the framework root)
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Research current (2025) White House communications officials and White House press corps reporters from major news organizations",
        parent=root,
        critical=True
    )

    # Administration officials (critical)
    admin_officials = evaluator.add_parallel(
        id="administration_officials",
        desc="Identify and provide details about current White House and State Department communications officials",
        parent=task_root,
        critical=True
    )

    # Press Secretary subtree
    await verify_press_secretary(
        evaluator=evaluator,
        parent_node=admin_officials,
        ps=extracted.press_secretary
    )

    # State Department Spokesperson subtree
    await verify_state_spokesperson(
        evaluator=evaluator,
        parent_node=admin_officials,
        spox=extracted.state_spox
    )

    # Press corps (critical)
    press_corps = evaluator.add_parallel(
        id="press_corps_reporters",
        desc="Identify White House reporters from three major news organizations",
        parent=task_root,
        critical=True
    )

    reporters = extracted.reporters or PressCorps()

    # NYT
    await verify_reporter(
        evaluator=evaluator,
        parent_node=press_corps,
        rep=reporters.nyt,
        org_key="nyt"
    )

    # CNN
    await verify_reporter(
        evaluator=evaluator,
        parent_node=press_corps,
        rep=reporters.cnn,
        org_key="cnn"
    )

    # Politico
    await verify_reporter(
        evaluator=evaluator,
        parent_node=press_corps,
        rep=reporters.politico,
        org_key="politico"
    )

    # Return summary
    return evaluator.get_summary()