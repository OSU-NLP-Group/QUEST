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
TASK_ID = "provost_2026_r1"
TASK_DESCRIPTION = (
    "Identify a major research university in the United States classified as R1 (doctoral university with very high research activity) "
    "that had an active provost search, announced a provost appointment, or filled its provost position in 2026. For the identified institution, "
    "research and provide the following information:\n\n"
    "1. Institution and Position: The name of the R1 university and verification that it had provost-related activity (search, appointment, or hiring) "
    "specifically in 2026. Include the name of the current or newly appointed provost if available.\n\n"
    "2. Educational Requirements: The terminal degree requirements for the provost position (e.g., PhD, EdD, or equivalent), including whether specific "
    "academic fields are required or preferred.\n\n"
    "3. Administrative Experience Requirements: The prior administrative experience requirements, including the type(s) of experience and whether a minimum number "
    "of years is specified.\n\n"
    "4. Dean-Level Experience Path: Whether prior dean-level or equivalent senior academic leadership experience is required, preferred, or typically expected.\n\n"
    "5. Compensation Information: If publicly available, the salary range or base compensation for the provost position.\n\n"
    "For each piece of information, provide a reference URL from official university sources, credible news outlets, or authoritative higher education publications."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class InstitutionAndActivity(BaseModel):
    institution_name: Optional[str] = None
    us_sources: List[str] = Field(default_factory=list)
    r1_sources: List[str] = Field(default_factory=list)
    provost_activity_type: Optional[str] = None  # search | appointment | hiring | other
    provost_activity_year: Optional[str] = None  # e.g., "2026"
    provost_activity_description: Optional[str] = None
    provost_activity_sources: List[str] = Field(default_factory=list)
    provost_name: Optional[str] = None
    provost_name_sources: List[str] = Field(default_factory=list)


class EducationBlock(BaseModel):
    terminal_degree_requirement: Optional[str] = None  # e.g., "Ph.D.", "Ed.D.", "earned doctorate", "terminal degree"
    terminal_degree_sources: List[str] = Field(default_factory=list)
    field_requirement_status: Optional[str] = None  # e.g., "required", "preferred", "any field", "not specified"
    field_requirement_sources: List[str] = Field(default_factory=list)


class AdminExperienceBlock(BaseModel):
    progressive_required_status: Optional[str] = None  # e.g., "required", "preferred", "not specified"
    progressive_sources: List[str] = Field(default_factory=list)
    admin_experience_types: List[str] = Field(default_factory=list)  # e.g., ["dean", "associate provost", "chair"]
    admin_experience_types_sources: List[str] = Field(default_factory=list)
    minimum_years: Optional[str] = None  # e.g., "10 years", "7+ years", "not specified"
    minimum_years_sources: List[str] = Field(default_factory=list)


class DeanExperienceBlock(BaseModel):
    dean_level_status: Optional[str] = None  # e.g., "required", "preferred", "typical/expected", "not specified"
    dean_level_sources: List[str] = Field(default_factory=list)


class CompensationBlock(BaseModel):
    compensation_text: Optional[str] = None  # e.g., "$400,000 - $550,000", "not publicly disclosed"
    compensation_sources: List[str] = Field(default_factory=list)


class ProvostResearchExtraction(BaseModel):
    institution: Optional[InstitutionAndActivity] = None
    education: Optional[EducationBlock] = None
    admin: Optional[AdminExperienceBlock] = None
    dean: Optional[DeanExperienceBlock] = None
    compensation: Optional[CompensationBlock] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_overall() -> str:
    return """
Extract the requested information exactly as stated in the assistant's answer. Do not invent anything. If something is not present in the answer, return null (or an empty list for URL fields). If multiple institutions are mentioned, extract only the first one that is addressed with citations.

Return a JSON object conforming to the following schema (null for any missing field; empty arrays for missing URL lists):

- institution:
  - institution_name: string | null
  - us_sources: array of URLs that support the institution being in the United States (may include about/contact pages, official pages, Wikipedia if clearly US, credible pages cited in the answer)
  - r1_sources: array of URLs that support R1 (Carnegie "Very high research activity") classification
  - provost_activity_type: string | null  (use one of: "search", "appointment", "hiring", or "other" if not one of the above)
  - provost_activity_year: string | null (e.g., "2026" if explicitly stated or implied in the answer)
  - provost_activity_description: string | null (a short phrase summarizing the activity from the answer)
  - provost_activity_sources: array of URLs cited for the 2026 provost activity
  - provost_name: string | null (if the answer supplies a specific person’s name tied to the 2026 provost activity)
  - provost_name_sources: array of URLs cited for the provost’s name (can be the same as activity sources if applicable)

- education:
  - terminal_degree_requirement: string | null (e.g., "Ph.D.", "Ed.D.", "earned doctorate", "terminal degree", "doctorate or equivalent")
  - terminal_degree_sources: array of URLs cited for terminal degree requirement
  - field_requirement_status: string | null (e.g., "required", "preferred", "any field", "not specified")
  - field_requirement_sources: array of URLs cited for field requirement status

- admin:
  - progressive_required_status: string | null (e.g., "required", "preferred", "not specified")
  - progressive_sources: array of URLs cited for the progressive administrative experience requirement
  - admin_experience_types: array of strings (e.g., ["dean", "associate provost", "department chair"]; if not specified, return an empty array)
  - admin_experience_types_sources: array of URLs cited for types of experience
  - minimum_years: string | null (e.g., "10 years", "7+ years", "not specified")
  - minimum_years_sources: array of URLs cited for minimum years requirement

- dean:
  - dean_level_status: string | null (e.g., "required", "preferred", "typical/expected", "not specified")
  - dean_level_sources: array of URLs cited for dean-level experience status

- compensation:
  - compensation_text: string | null (salary/range if disclosed; if the answer explicitly says it's not disclosed, put "not publicly disclosed")
  - compensation_sources: array of URLs cited for compensation info (if not disclosed, cite the relevant page used to conclude that)

IMPORTANT:
- For every field that states "with citation", populate the corresponding URL list(s) with exactly the URLs present in the answer. If the answer provided no URL for that field, return an empty list for that URL field.
- Preserve the exact phrasing present in the answer for all text fields where possible (e.g., "terminal degree", "earned doctorate").
- Do not infer URLs; only include URLs explicitly present in the answer (plain links or markdown links).
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            u2 = u.strip()
            if not u2 or u2 in seen:
                continue
            seen.add(u2)
            out.append(u2)
    return out


def _safe(val: Optional[str], default: str) -> str:
    return val.strip() if isinstance(val, str) and val.strip() else default


def _require_urls_instruction(base_instruction: str, urls: List[str]) -> str:
    if urls:
        return (
            base_instruction
            + "\nYou must base your judgment solely on the provided URL evidence. "
              "If the pages do not clearly support the claim, answer Incorrect."
        )
    else:
        return (
            base_instruction
            + "\nNo URLs were provided for this claim. According to the evaluation policy, "
              "you must judge the claim Incorrect due to lack of source evidence, regardless of the answer text."
        )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_institution_section(evaluator: Evaluator, parent) -> None:
    section = evaluator.add_parallel(
        id="Institution_And_2026_Provost_Activity",
        desc="Identify the institution and verify it meets location/R1/2026 provost-activity requirements; provide provost name if available.",
        parent=parent,
        critical=True,
    )

    data = evaluator.find_node("root")  # placeholder to show we can access evaluator if needed (not used directly)
    # Pull extracted data from the already recorded extraction (last extraction result)
    # Safer approach: pass the extracted object into this function; but we will fetch from closure in main.

    # We will store the extracted object on the evaluator as a custom info to re-access.
    # Instead, we return here; actual fields will be passed via closure in evaluate function.
    # This function is only a container name; real logic is implemented in the variant below.
    return


async def verify_institution_and_activity(
    evaluator: Evaluator,
    parent,
    ex: ProvostResearchExtraction,
) -> None:
    inst = ex.institution or InstitutionAndActivity()

    section = evaluator.add_parallel(
        id="Institution_And_2026_Provost_Activity",
        desc="Identify the institution and verify it meets location/R1/2026 provost-activity requirements; provide provost name if available.",
        parent=parent,
        critical=True,
    )

    # 1) US Institution
    us_node = evaluator.add_leaf(
        id="US_Institution_With_Citation",
        desc="The identified institution is located in the United States, supported by a reference URL.",
        parent=section,
        critical=True,
    )
    inst_name = _safe(inst.institution_name, "the identified institution")
    us_sources = _norm_sources(inst.us_sources)
    us_claim = f"The institution named '{inst_name}' is located in the United States."
    await evaluator.verify(
        claim=us_claim,
        node=us_node,
        sources=us_sources,
        additional_instruction=_require_urls_instruction(
            "Accept evidence that clearly shows a U.S. location (e.g., references to a U.S. state or 'United States').",
            us_sources,
        ),
    )

    # 2) R1 Classification
    r1_node = evaluator.add_leaf(
        id="R1_Classification_With_Citation",
        desc="The identified institution is classified as Carnegie R1 (doctoral university with very high research activity), supported by a reference URL.",
        parent=section,
        critical=True,
    )
    r1_sources = _norm_sources(inst.r1_sources)
    r1_claim = (
        f"The institution '{inst_name}' is classified as Carnegie R1 (Doctoral Universities – Very high research activity)."
    )
    await evaluator.verify(
        claim=r1_claim,
        node=r1_node,
        sources=r1_sources,
        additional_instruction=_require_urls_instruction(
            "Accept synonyms like 'very high research activity' or 'R1'. Do not accept R2 or other categories.",
            r1_sources,
        ),
    )

    # 3) Provost activity in 2026
    act_node = evaluator.add_leaf(
        id="Provost_Activity_In_2026_With_Citation",
        desc="The institution had an active provost search, announced a provost appointment, or filled its provost position specifically in 2026, supported by a reference URL.",
        parent=section,
        critical=True,
    )
    act_sources = _norm_sources(inst.provost_activity_sources)
    act_type = _safe(inst.provost_activity_type, "provost-related activity")
    act_year = _safe(inst.provost_activity_year, "2026")
    act_claim = (
        f"In {act_year}, the institution '{inst_name}' had a {act_type} related to the provost position."
    )
    await evaluator.verify(
        claim=act_claim,
        node=act_node,
        sources=act_sources,
        additional_instruction=_require_urls_instruction(
            "The page(s) must clearly indicate the year 2026 (e.g., publication date or explicit mention) for a provost search/appointment/hire. "
            "References to 2025 or 2024 alone are not sufficient.",
            act_sources,
        ),
    )

    # 4) Provost name if available
    name_node = evaluator.add_leaf(
        id="Provost_Name_If_Available_With_Citation",
        desc="Provide the current/newly appointed provost name if available in authoritative sources; otherwise explicitly state it is not available from consulted authoritative sources. Include a reference URL supporting the name (if provided) or supporting the underlying 2026 provost activity document(s) consulted.",
        parent=section,
        critical=True,
    )
    # Use name sources if provided; otherwise fall back to activity sources (per rubric).
    name_sources = _norm_sources(inst.provost_name_sources, inst.provost_activity_sources)
    if inst.provost_name and inst.provost_name.strip():
        nm = inst.provost_name.strip()
        name_claim = (
            f"The provided sources state that {nm} is (or was appointed as) the provost (or equivalent title) at '{inst_name}' in connection with the 2026 activity."
        )
        add_ins = _require_urls_instruction(
            "Accept close title variants like 'Provost and Executive Vice President for Academic Affairs'. "
            "The page must clearly associate the person with the provost role.",
            name_sources,
        )
    else:
        nm = "not provided"
        name_claim = (
            f"The provided sources document the 2026 provost-related activity at '{inst_name}' but do not clearly provide the provost's personal name."
        )
        add_ins = _require_urls_instruction(
            "Mark Correct only if the pages support the 2026 provost-related activity and do not clearly state a specific individual's name as provost. "
            "If the pages clearly name an individual as provost, this claim is Incorrect.",
            name_sources,
        )
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=name_sources,
        additional_instruction=add_ins,
    )


async def verify_education(
    evaluator: Evaluator,
    parent,
    ex: ProvostResearchExtraction,
) -> None:
    edu = ex.education or EducationBlock()

    section = evaluator.add_parallel(
        id="Educational_Requirements",
        desc="Report terminal-degree and field requirements for the provost position, with citations.",
        parent=parent,
        critical=True,
    )

    # Terminal degree requirement
    term_node = evaluator.add_leaf(
        id="Terminal_Degree_Requirement_With_Citation",
        desc="State the terminal degree requirement for the provost role (e.g., PhD/EdD/equivalent) and any stated accreditation requirement, supported by a reference URL.",
        parent=section,
        critical=True,
    )
    term_sources = _norm_sources(edu.terminal_degree_sources)
    term_req = _safe(edu.terminal_degree_requirement, "terminal degree (doctorate or equivalent)")
    term_claim = (
        f"The provost position requires a terminal degree consistent with the answer (e.g., '{term_req}' or equivalent), as evidenced by the provided sources."
    )
    await evaluator.verify(
        claim=term_claim,
        node=term_node,
        sources=term_sources,
        additional_instruction=_require_urls_instruction(
            "Accept synonymous phrases like 'earned doctorate', 'terminal degree', 'Ph.D./Ed.D. or equivalent'.",
            term_sources,
        ),
    )

    # Academic field requirement status
    field_node = evaluator.add_leaf(
        id="Academic_Field_Requirement_Status_With_Citation",
        desc='State whether specific academic fields are required or preferred for the terminal degree; if not specified, state "not specified". Support with a reference URL.',
        parent=section,
        critical=True,
    )
    field_sources = _norm_sources(edu.field_requirement_sources)
    field_status = _safe(edu.field_requirement_status, "not specified")
    field_claim = (
        f"The sources indicate the status of specific academic field requirements for the provost's terminal degree as: '{field_status}'."
    )
    await evaluator.verify(
        claim=field_claim,
        node=field_node,
        sources=field_sources,
        additional_instruction=_require_urls_instruction(
            "If the posting/search materials do not mention specific fields, 'not specified' is acceptable; "
            "otherwise, they should indicate whether any fields are required or preferred.",
            field_sources,
        ),
    )


async def verify_admin_experience(
    evaluator: Evaluator,
    parent,
    ex: ProvostResearchExtraction,
) -> None:
    adm = ex.admin or AdminExperienceBlock()

    section = evaluator.add_parallel(
        id="Administrative_Experience_Requirements",
        desc="Report administrative experience requirements for the provost position, with citations.",
        parent=parent,
        critical=True,
    )

    # Progressive higher-ed admin experience required (not merely preferred)
    prog_node = evaluator.add_leaf(
        id="Progressive_HigherEd_Admin_Experience_Required_With_Citation",
        desc="Verify that the provost position requires demonstrated progressive administrative experience in higher education (i.e., not merely desirable/optional), supported by a reference URL.",
        parent=section,
        critical=True,
    )
    prog_sources = _norm_sources(adm.progressive_sources)
    prog_claim = (
        "The provided sources explicitly require (not merely prefer) demonstrated progressive higher education administrative experience "
        "for the provost role."
    )
    await evaluator.verify(
        claim=prog_claim,
        node=prog_node,
        sources=prog_sources,
        additional_instruction=_require_urls_instruction(
            "Accept phrases like 'progressively responsible leadership in higher education is required'. "
            "If the sources only say 'preferred', 'desired', or do not state 'required', mark Incorrect.",
            prog_sources,
        ),
    )

    # Types of administrative experience
    types_node = evaluator.add_leaf(
        id="Administrative_Experience_Types_With_Citation",
        desc='Identify the types of administrative roles/experience required or preferred (e.g., dean, associate provost, department chair, etc.); if not specified, state "not specified". Support with a reference URL.',
        parent=section,
        critical=True,
    )
    types_sources = _norm_sources(adm.admin_experience_types_sources)
    roles_list = adm.admin_experience_types if adm.admin_experience_types else []
    roles_text = ", ".join(roles_list) if roles_list else "not specified"
    types_claim = (
        f"The sources indicate the following administrative role types for the provost candidates: {roles_text}."
    )
    await evaluator.verify(
        claim=types_claim,
        node=types_node,
        sources=types_sources,
        additional_instruction=_require_urls_instruction(
            "Verify that the listed role types (if any) appear on the provided pages. "
            "If the posting does not specify role types, 'not specified' is acceptable.",
            types_sources,
        ),
    )

    # Minimum years of administrative experience
    years_node = evaluator.add_leaf(
        id="Minimum_Years_Admin_Experience_With_Citation",
        desc='Determine whether a minimum number of years of administrative experience is specified; if specified, report the number; if not, state "not specified". Support with a reference URL.',
        parent=section,
        critical=True,
    )
    years_sources = _norm_sources(adm.minimum_years_sources)
    years_text = _safe(adm.minimum_years, "not specified")
    years_claim = (
        f"The sources {'specify a minimum of ' + years_text if years_text.lower() != 'not specified' else 'do not specify a minimum years requirement'} "
        "for administrative experience."
    )
    await evaluator.verify(
        claim=years_claim,
        node=years_node,
        sources=years_sources,
        additional_instruction=_require_urls_instruction(
            "Accept reasonable numeric formats (e.g., '10 years', '7+ years'). If no minimum is stated, 'not specified' is acceptable.",
            years_sources,
        ),
    )


async def verify_dean_path(
    evaluator: Evaluator,
    parent,
    ex: ProvostResearchExtraction,
) -> None:
    dean = ex.dean or DeanExperienceBlock()

    section = evaluator.add_parallel(
        id="Dean_Level_Experience_Path",
        desc="Report whether dean-level (or equivalent senior academic leadership) experience is required, preferred, or typically expected, with citations.",
        parent=parent,
        critical=True,
    )

    dean_node = evaluator.add_leaf(
        id="Dean_Level_Status_With_Citation",
        desc="State whether dean-level/equivalent senior academic leadership experience is required, preferred, or described as typical/expected for provost candidates (based on the institution’s materials for the search/appointment and/or authoritative higher-ed sources tied to that search). Support with a reference URL.",
        parent=section,
        critical=True,
    )
    dean_sources = _norm_sources(dean.dean_level_sources)
    dean_status = _safe(dean.dean_level_status, "not specified")
    dean_claim = (
        f"The sources indicate the status for dean-level or equivalent senior academic leadership experience as: '{dean_status}'."
    )
    await evaluator.verify(
        claim=dean_claim,
        node=dean_node,
        sources=dean_sources,
        additional_instruction=_require_urls_instruction(
            "Accept titles equivalent to dean-level (e.g., 'dean', 'vice provost', 'associate provost', 'executive vice president for academic affairs') "
            "when clearly framed as senior academic leadership experience.",
            dean_sources,
        ),
    )


async def verify_compensation(
    evaluator: Evaluator,
    parent,
    ex: ProvostResearchExtraction,
) -> None:
    comp = ex.compensation or CompensationBlock()

    section = evaluator.add_parallel(
        id="Compensation_Information",
        desc="Report compensation if publicly disclosed; otherwise state it is not publicly disclosed, with citations.",
        parent=parent,
        critical=True,
    )

    comp_node = evaluator.add_leaf(
        id="Compensation_Disclosure_And_Value_With_Citation",
        desc="If a salary range/base/compensation is publicly disclosed, report it; otherwise explicitly state compensation is not publicly disclosed. Support with a reference URL and do not provide speculative/unsourced numbers.",
        parent=section,
        critical=True,
    )
    comp_sources = _norm_sources(comp.compensation_sources)
    comp_text = _safe(comp.compensation_text, "not publicly disclosed")
    if comp_text.lower() == "not publicly disclosed":
        comp_claim = "The sources indicate that compensation for the provost position is not publicly disclosed."
        add_ins = _require_urls_instruction(
            "Mark Correct only if the pages lack a disclosed salary figure/range for the provost role, or explicitly state it is not disclosed.",
            comp_sources,
        )
    else:
        comp_claim = f"The sources disclose provost compensation or salary information (for example: '{comp_text}')."
        add_ins = _require_urls_instruction(
            "The disclosed value/range must be explicitly present on the provided page(s). Do not accept speculative or unrelated figures.",
            comp_sources,
        )
    await evaluator.verify(
        claim=comp_claim,
        node=comp_node,
        sources=comp_sources,
        additional_instruction=add_ins,
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the 'provost_2026_r1' task and return a structured result dictionary.
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

    # Extraction
    extraction: ProvostResearchExtraction = await evaluator.extract(
        prompt=prompt_extract_overall(),
        template_class=ProvostResearchExtraction,
        extraction_name="provost_research_extraction",
    )

    # Add a critical top-level node representing the overall rubric root
    top = evaluator.add_parallel(
        id="Provost_Career_Research",
        desc="Identify one U.S. Carnegie R1 university with provost-related activity in 2026 and report requested attributes with verifiable citations.",
        parent=root,
        critical=True,
    )

    # Build and verify each rubric section
    await verify_institution_and_activity(evaluator, top, extraction)
    await verify_education(evaluator, top, extraction)
    await verify_admin_experience(evaluator, top, extraction)
    await verify_dean_path(evaluator, top, extraction)
    await verify_compensation(evaluator, top, extraction)

    # Return summary
    return evaluator.get_summary()