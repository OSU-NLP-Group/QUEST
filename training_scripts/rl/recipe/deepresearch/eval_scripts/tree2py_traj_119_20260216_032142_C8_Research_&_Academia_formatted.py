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
TASK_ID = "marine_postdoc_programs_ma_wh"
TASK_DESCRIPTION = (
    "Identify three postdoctoral fellowship programs in oceanography or marine biology that are available to researchers affiliated with institutions in the Woods Hole, Massachusetts area. "
    "For each program, provide: name, sponsoring organization, confirmation it is a postdoctoral fellowship, the specific field (oceanography/marine biology/related), "
    "citizenship/residency requirement, doctoral degree status requirement, research proposal page limit, number of recommendation letters, duration, stipend/award amount, "
    "primary research focus areas, application deadline or submission window, and an official reference URL. Ensure all three programs are distinct and each meets the specified criteria."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    program_name: Optional[str] = None
    organization: Optional[str] = None
    program_type: Optional[str] = None
    field_area: Optional[str] = None
    ma_availability: Optional[str] = None
    citizenship_requirement: Optional[str] = None
    degree_requirement: Optional[str] = None
    proposal_page_limit: Optional[str] = None
    letters_required: Optional[str] = None
    duration: Optional[str] = None
    funding_amount: Optional[str] = None
    research_focus: Optional[str] = None
    deadline: Optional[str] = None
    url: Optional[str] = None


class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract up to five postdoctoral fellowship programs mentioned in the answer, capturing the following fields for each program exactly as written in the answer:
    - program_name: The program name.
    - organization: Sponsoring organization or institution.
    - program_type: The stated program type (e.g., "postdoctoral fellowship", "postdoctoral scholar program").
    - field_area: The field or research area as described.
    - ma_availability: Any statement about availability to researchers affiliated with institutions in Massachusetts or the Woods Hole area, or a statement indicating no restrictive geographic eligibility (e.g., open nationally or internationally).
    - citizenship_requirement: Eligibility regarding citizenship or residency status.
    - degree_requirement: Eligibility regarding doctoral degree status (e.g., PhD required, or must have PhD by start date).
    - proposal_page_limit: The research proposal page limit for the application (if stated).
    - letters_required: The number of recommendation letters required (if stated).
    - duration: The fellowship duration (e.g., months or years, or a range).
    - funding_amount: The stipend or award amount (or range).
    - research_focus: Primary research focus areas or eligible research topics.
    - deadline: The application deadline or submission window.
    - url: A reference URL to the official program information. Extract a complete http/https URL if present.

    GENERAL RULES:
    - Do not infer or invent missing information; if a field is not provided in the answer, set it to null.
    - Keep values as strings, preserving formatting (e.g., ranges like "1-2 years" or "2–3 letters").
    - Only extract actual URLs that appear in the answer; if missing, set to null.
    - Return a JSON with a top-level 'programs' array of objects. Each object must include all fields listed above with null for any missing value.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _is_nonempty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _is_valid_url(s: Optional[str]) -> bool:
    return _is_nonempty(s) and (s.strip().lower().startswith("http://") or s.strip().lower().startswith("https://"))


# --------------------------------------------------------------------------- #
# Verification for a single program                                           #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    parent_node,
    prog: ProgramItem,
    idx: int,
) -> None:
    """
    Build verification subtree for one program and run checks.
    All content checks are critical within the program; the program node itself is non-critical to allow partial credit across programs.
    """
    pnum = idx + 1
    program_node = evaluator.add_parallel(
        id=f"Program_{pnum}",
        desc=f"Postdoctoral fellowship program #{pnum} with required information and marine/ocean field relevance",
        parent=parent_node,
        critical=False
    )

    # Existence checks (critical within program)
    name_exists = evaluator.add_custom_node(
        result=_is_nonempty(prog.program_name),
        id=f"Program_{pnum}_Name",
        desc=f"Program name is provided",
        parent=program_node,
        critical=True
    )
    org_exists = evaluator.add_custom_node(
        result=_is_nonempty(prog.organization),
        id=f"Program_{pnum}_Organization",
        desc=f"Sponsoring organization or institution is identified",
        parent=program_node,
        critical=True
    )
    url_exists = evaluator.add_custom_node(
        result=_is_valid_url(prog.url),
        id=f"Program_{pnum}_URL_Reference",
        desc=f"A valid reference URL to official program information is included",
        parent=program_node,
        critical=True
    )

    # Build a handy label
    prog_label = prog.program_name if _is_nonempty(prog.program_name) else f"Program #{pnum}"
    url_source = prog.url if _is_valid_url(prog.url) else None

    # Type verification: postdoctoral fellowship
    type_leaf = evaluator.add_leaf(
        id=f"Program_{pnum}_Type_Verification",
        desc=f"Program is confirmed to be a postdoctoral fellowship (not PhD or master's program)",
        parent=program_node,
        critical=True
    )
    type_claim = "This program is a postdoctoral fellowship (postdoctoral research/scholar program), not a PhD or master's student degree program."
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=url_source,
        extra_prerequisites=[url_exists],
        additional_instruction="Accept synonyms like 'postdoctoral scholar program', 'postdoctoral research fellowship', or 'postdoctoral program'. Reject graduate student-only programs."
    )

    # Field relevance: oceanography/marine biology/related marine science
    field_leaf = evaluator.add_leaf(
        id=f"Program_{pnum}_Field_Relevance",
        desc=f"Program is in oceanography, marine biology, biological oceanography, or related marine science field",
        parent=program_node,
        critical=True
    )
    field_claim = (
        "The program focuses on oceanography or marine biology, or a closely related marine science field "
        "(e.g., biological oceanography, physical oceanography, marine ecology, marine chemistry, marine geophysics)."
    )
    await evaluator.verify(
        claim=field_claim,
        node=field_leaf,
        sources=url_source,
        extra_prerequisites=[url_exists],
        additional_instruction="Confirm that the program's scope explicitly relates to ocean or marine sciences. Allow closely related subfields within oceanography or marine biology."
    )

    # Massachusetts/Woods Hole availability
    ma_leaf = evaluator.add_leaf(
        id=f"Program_{pnum}_MA_Affiliation",
        desc=f"Program is available to researchers at Massachusetts institutions or Woods Hole area",
        parent=program_node,
        critical=True
    )
    ma_claim = (
        "The program is available to researchers affiliated with institutions in Massachusetts or the Woods Hole area. "
        "This is satisfied if eligibility is nationwide (US) or international without excluding Massachusetts, "
        "or if the host/eligible institutions include Massachusetts institutions (e.g., Woods Hole Oceanographic Institution, Marine Biological Laboratory, MIT-WHOI)."
    )
    await evaluator.verify(
        claim=ma_claim,
        node=ma_leaf,
        sources=url_source,
        extra_prerequisites=[url_exists],
        additional_instruction="If the page shows the program is open broadly (national or international) or hosted in Massachusetts (e.g., WHOI/MBL), consider it available to MA/Woods Hole researchers. Fail if eligibility explicitly excludes US/MA researchers."
    )

    # Citizenship/residency requirement
    citizen_leaf = evaluator.add_leaf(
        id=f"Program_{pnum}_Citizenship_Requirement",
        desc=f"Eligibility requirement regarding citizenship or residency status is stated",
        parent=program_node,
        critical=True
    )
    citizen_text = prog.citizenship_requirement or ""
    citizen_claim = f"The program's citizenship/residency eligibility is: {citizen_text}"
    await evaluator.verify(
        claim=citizen_claim,
        node=citizen_leaf,
        sources=url_source,
        extra_prerequisites=[url_exists],
        additional_instruction="Confirm text about citizenship or residency eligibility (e.g., open to all nationalities, U.S. citizens/permanent residents only, etc.). Allow equivalent wording."
    )

    # Doctoral degree requirement/status
    degree_leaf = evaluator.add_leaf(
        id=f"Program_{pnum}_Degree_Requirement",
        desc=f"Doctoral degree requirement or status is specified",
        parent=program_node,
        critical=True
    )
    degree_text = prog.degree_requirement or ""
    degree_claim = f"The program's doctoral degree requirement/status is: {degree_text}"
    await evaluator.verify(
        claim=degree_claim,
        node=degree_leaf,
        sources=url_source,
        extra_prerequisites=[url_exists],
        additional_instruction="Check if a PhD is required or must be completed by the start date, or equivalent statements."
    )

    # Proposal page limit
    proposal_leaf = evaluator.add_leaf(
        id=f"Program_{pnum}_Proposal_Page_Limit",
        desc=f"Research proposal page limit for application is provided",
        parent=program_node,
        critical=True
    )
    proposal_text = prog.proposal_page_limit or ""
    proposal_claim = f"The research proposal page limit is: {proposal_text}"
    await evaluator.verify(
        claim=proposal_claim,
        node=proposal_leaf,
        sources=url_source,
        extra_prerequisites=[url_exists],
        additional_instruction="Look for instructions specifying a maximum number of pages for the research proposal; allow variants like excluding references or single/double-spaced."
    )

    # Letters of recommendation required
    letters_leaf = evaluator.add_leaf(
        id=f"Program_{pnum}_Letters_Required",
        desc=f"Number of required recommendation letters is specified",
        parent=program_node,
        critical=True
    )
    letters_text = prog.letters_required or ""
    letters_claim = f"The number of recommendation letters required is: {letters_text}"
    await evaluator.verify(
        claim=letters_claim,
        node=letters_leaf,
        sources=url_source,
        extra_prerequisites=[url_exists],
        additional_instruction="Allow phrasing like 'two or three letters' or 'up to three'."
    )

    # Fellowship duration
    duration_leaf = evaluator.add_leaf(
        id=f"Program_{pnum}_Fellowship_Duration",
        desc=f"Duration of the fellowship in months or years is stated",
        parent=program_node,
        critical=True
    )
    duration_text = prog.duration or ""
    duration_claim = f"The fellowship duration is: {duration_text}"
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=url_source,
        extra_prerequisites=[url_exists],
        additional_instruction="Check for duration length (e.g., 12 months, 18–24 months, up to 2 years) including ranges or renewals."
    )

    # Funding amount / stipend
    funding_leaf = evaluator.add_leaf(
        id=f"Program_{pnum}_Funding_Amount",
        desc=f"Stipend or award amount (or funding range) is provided",
        parent=program_node,
        critical=True
    )
    funding_text = prog.funding_amount or ""
    funding_claim = f"The stipend/award amount or range is: {funding_text}"
    await evaluator.verify(
        claim=funding_claim,
        node=funding_leaf,
        sources=url_source,
        extra_prerequisites=[url_exists],
        additional_instruction="Verify stipend or award amount; allow ranges, approximate values, and total support descriptions."
    )

    # Research focus areas
    focus_leaf = evaluator.add_leaf(
        id=f"Program_{pnum}_Research_Focus",
        desc=f"Primary research focus area or eligible research topics are described",
        parent=program_node,
        critical=True
    )
    focus_text = prog.research_focus or ""
    focus_claim = f"The primary research focus areas or eligible topics are: {focus_text}"
    await evaluator.verify(
        claim=focus_claim,
        node=focus_leaf,
        sources=url_source,
        extra_prerequisites=[url_exists],
        additional_instruction="Confirm the listed marine/ocean-related research themes or eligible topics; allow paraphrase and synonymous terms."
    )

    # Application deadline / window
    deadline_leaf = evaluator.add_leaf(
        id=f"Program_{pnum}_Application_Deadline",
        desc=f"Application deadline or submission window is provided",
        parent=program_node,
        critical=True
    )
    deadline_text = prog.deadline or ""
    deadline_claim = f"The application deadline or submission window is: {deadline_text}"
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_leaf,
        sources=url_source,
        extra_prerequisites=[url_exists],
        additional_instruction="Confirm deadline date(s) or application window (e.g., annual cycle, rolling deadline, or specific date)."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the marine/oceanography postdoctoral programs task.
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
        default_model=model
    )

    # Extract programs from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction"
    )

    # Keep first 3 programs; pad with empty if fewer
    programs: List[ProgramItem] = (extracted.programs or [])[:3]
    while len(programs) < 3:
        programs.append(ProgramItem())

    evaluator.add_custom_info(
        info={"extracted_program_count": len(extracted.programs or []), "used_programs": 3},
        info_type="extraction_stats"
    )

    # Build three program subtrees in parallel under root
    tasks = []
    for i in range(3):
        tasks.append(verify_program(evaluator, root, programs[i], i))
    for t in tasks:
        await t

    return evaluator.get_summary()