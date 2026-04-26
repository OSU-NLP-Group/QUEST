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
TASK_ID = "va_president_edd_program_director_2025"
TASK_DESCRIPTION = (
    "In December 2025, a major public university in Virginia appointed a new president who had been serving as the dean "
    "of its business school since August 2015. Prior to his academic leadership career, this individual worked for an "
    "international consulting firm for 26 years. He earned a Doctor of Education (EdD) degree in Higher Education "
    "Management in 2015 from a specific university's graduate school of education program. What is the full name of the "
    "current director (as of the 2025-2026 academic year) of that Executive Doctorate in Higher Education Management "
    "program, when did they assume this director role in 2025, and what was one of their prior professional positions or "
    "affiliations before taking on this directorship?"
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PresidentAnchorExtraction(BaseModel):
    # Identity and appointment anchoring
    president_name: Optional[str] = None
    university: Optional[str] = None
    appointment_date: Optional[str] = None  # e.g., "December 5, 2025" or "December 2025"
    business_school_dean_since: Optional[str] = None  # e.g., "since August 2015"
    consulting_firm: Optional[str] = None  # e.g., "McKinsey & Company"
    consulting_years: Optional[str] = None  # e.g., "26" or "26 years"

    # EdD degree context (for linking the correct program)
    edd_program_name: Optional[str] = None  # "Executive Doctorate in Higher Education Management"
    edd_institution: Optional[str] = None  # e.g., "University of Pennsylvania"
    edd_school: Optional[str] = None  # e.g., "Graduate School of Education"
    edd_year: Optional[str] = None  # "2015"

    # URLs mentioned in the answer for these facts
    sources_president: List[str] = Field(default_factory=list)  # appointment/dean/consulting
    sources_edd: List[str] = Field(default_factory=list)  # degree/program-related citations


class ProgramDirectorExtraction(BaseModel):
    # Program identification
    program_name: Optional[str] = None  # Executive Doctorate in Higher Education Management
    institution: Optional[str] = None  # University name
    school: Optional[str] = None  # Graduate School of Education (or equivalent)
    program_urls: List[str] = Field(default_factory=list)  # program pages cited

    # Director details (as of AY 2025–2026)
    director_full_name: Optional[str] = None
    director_assumption_date_2025: Optional[str] = None  # month and year (or date) in 2025
    director_prior_position: Optional[str] = None
    director_urls: List[str] = Field(default_factory=list)  # director-specific citations


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_president_anchor() -> str:
    return """
    Extract from the answer the anchoring details about the referenced university president and their EdD program. 
    You must only extract information explicitly present in the answer. Provide null for any missing field.

    Required fields:
    - president_name: Full name of the referenced president (if provided).
    - university: The university to which they were appointed as president.
    - appointment_date: The appointment/announcement date for the presidency (e.g., 'December 2025' or specific date).
    - business_school_dean_since: The fact that the individual had been serving as dean of the university’s business school since August 2015 (capture the phrasing or 'since August 2015').
    - consulting_firm: The international consulting firm’s name (if provided).
    - consulting_years: The number of years (e.g., '26 years') the individual worked for that consulting firm.
    - edd_program_name: Name of the EdD program (should be Executive Doctorate in Higher Education Management or equivalent phrasing).
    - edd_institution: The university awarding the EdD.
    - edd_school: The specific graduate school of education within that institution.
    - edd_year: The EdD year (should be 2015).

    Also extract the URLs mentioned in the answer that support these facts:
    - sources_president: A list of URLs supporting appointment, dean since Aug 2015, and consulting firm experience.
    - sources_edd: A list of URLs supporting the EdD program and the 2015 EdD fact.

    IMPORTANT:
    - Only include URLs actually present in the answer text.
    - If a field is not explicitly mentioned, set it to null.
    """


def prompt_extract_program_director() -> str:
    return """
    Extract from the answer the Executive Doctorate in Higher Education Management (EdD) program identification 
    and the current director details for the 2025–2026 academic year. Only extract information explicitly present in the answer. 
    Provide null for any missing field.

    Required fields:
    - program_name: The program’s official name as given in the answer.
    - institution: The university that offers this program.
    - school: The specific graduate school of education (or equivalent) that houses this program.
    - program_urls: A list of URLs in the answer that point to the program’s official pages or relevant information.

    - director_full_name: The full name of the current program director for the 2025–2026 academic year.
    - director_assumption_date_2025: When the director assumed the role in 2025 (month and year or a specific date).
    - director_prior_position: One prior professional position or affiliation the director held before this role.
    - director_urls: A list of URLs cited in the answer that support director identity, assumption timing, and/or the prior position.

    IMPORTANT:
    - Only include URLs actually present in the answer text.
    - If any item is not explicitly mentioned, set it to null or an empty list as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _pick_sources(*url_lists: Optional[List[str]]) -> List[str]:
    """Pick the first non-empty list of URLs; otherwise return empty list."""
    for urls in url_lists:
        if urls and len(urls) > 0:
            return urls
    return []


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    pres: PresidentAnchorExtraction,
    prog: ProgramDirectorExtraction
) -> None:
    """
    Build the verification tree according to the rubric and run evidence-based checks.
    """

    # Top-level task node (Critical, Sequential)
    task_node = evaluator.add_sequential(
        id="task_completion",
        desc="Identify the Executive Doctorate in Higher Education Management program referenced via the described Virginia public-university president, then provide the director details for the 2025–2026 academic year.",
        parent=root_node,
        critical=True
    )

    # ---------------------------------------------------------
    # 1) Verify President Constraints (Parallel, Critical)
    # ---------------------------------------------------------
    pres_node = evaluator.add_parallel(
        id="verify_president_constraints",
        desc="Confirm the referenced president matches all identifying constraints given in the prompt/constraints (used to anchor the correct EdD program).",
        parent=task_node,
        critical=True
    )

    # Leaf: President appointed Dec 2025 to a major VA public university
    leaf_pres_appointed = evaluator.add_leaf(
        id="president_appointed_dec2025_va",
        desc="State that the president was appointed in December 2025 to a major public university in Virginia.",
        parent=pres_node,
        critical=True
    )

    pres_name = pres.president_name or "the referenced individual"
    uni = pres.university or "the university in question"
    appoint_date = pres.appointment_date or "December 2025"
    claim_appointed = (
        f"The cited sources state that {pres_name} was appointed (or named/announced as) President of {uni} in December 2025, "
        f"and that {uni} is a public university in Virginia."
    )
    await evaluator.verify(
        claim=claim_appointed,
        node=leaf_pres_appointed,
        sources=_pick_sources(pres.sources_president),
        additional_instruction=(
            "Verify that the appointment or announcement occurred in December 2025 (exact date not required as long as it's within December 2025). "
            "Accept phrasing such as 'appointed', 'named', or 'selected as the next president.' "
            "Also confirm that the institution is a public university in Virginia; if the page is an official university page clearly tied to Virginia, "
            "that is sufficient evidence."
        )
    )

    # Leaf: Dean since August 2015
    leaf_pres_dean = evaluator.add_leaf(
        id="president_dean_since_aug2015",
        desc="State that the president had been serving as dean of the university’s business school since August 2015.",
        parent=pres_node,
        critical=True
    )
    dean_since = pres.business_school_dean_since or "since August 2015"
    claim_dean = (
        f"The cited sources state that {pres_name} had been serving as the dean of the university’s business school {dean_since}."
    )
    await evaluator.verify(
        claim=claim_dean,
        node=leaf_pres_dean,
        sources=_pick_sources(pres.sources_president),
        additional_instruction=(
            "Focus on verifying the 'since August 2015' aspect. Equivalent phrasings like 'since August 2015' or 'from August 2015' are acceptable."
        )
    )

    # Leaf: Consulting firm for 26 years
    leaf_pres_consult = evaluator.add_leaf(
        id="president_consulting_26_years",
        desc="State that the president worked for an international consulting firm for 26 years before entering academic leadership.",
        parent=pres_node,
        critical=True
    )
    firm = pres.consulting_firm or "an international consulting firm"
    years_txt = pres.consulting_years or "26 years"
    claim_consult = (
        f"The cited sources state that {pres_name} worked for {firm} for {years_txt} prior to his academic leadership career."
    )
    await evaluator.verify(
        claim=claim_consult,
        node=leaf_pres_consult,
        sources=_pick_sources(pres.sources_president),
        additional_instruction=(
            "Allow minor wording variations such as 'a 26-year career at [firm].' The core is that the duration is 26 years at an international consulting firm."
        )
    )

    # Leaf: EdD in HEM in 2015
    leaf_pres_edd = evaluator.add_leaf(
        id="president_edd_2015",
        desc="State that the president earned an EdD in Higher Education Management in 2015.",
        parent=pres_node,
        critical=True
    )
    edd_prog = pres.edd_program_name or "the Executive Doctorate in Higher Education Management program"
    edd_inst = pres.edd_institution or "the relevant university"
    edd_year = pres.edd_year or "2015"
    claim_edd = (
        f"The cited sources state that {pres_name} earned an EdD in Higher Education Management in {edd_year} from {edd_inst}."
    )
    await evaluator.verify(
        claim=claim_edd,
        node=leaf_pres_edd,
        sources=_pick_sources(pres.sources_edd, pres.sources_president),
        additional_instruction=(
            "Verify both the degree (EdD in Higher Education Management) and the conferral year 2015. "
            "Accept explicit mentions on alumni bios or official announcements listing the credential."
        )
    )

    # ---------------------------------------------------------
    # 2) Identify the EdD Program (Parallel, Critical)
    # ---------------------------------------------------------
    program_node = evaluator.add_parallel(
        id="identify_edd_program",
        desc="Identify the doctoral program referenced (the Executive Doctorate in Higher Education Management at the relevant university’s graduate school of education).",
        parent=task_node,
        critical=True
    )

    # Leaf: Program Name
    leaf_prog_name = evaluator.add_leaf(
        id="program_name_check",
        desc="Identify the program as an Executive Doctorate in Higher Education Management (EdD).",
        parent=program_node,
        critical=True
    )
    extracted_program_name = prog.program_name or pres.edd_program_name or "Executive Doctorate in Higher Education Management"
    claim_prog_name = (
        f"The cited sources indicate that the relevant program is called '{extracted_program_name}', which is an Executive Doctorate in Higher Education Management (EdD)."
    )
    await evaluator.verify(
        claim=claim_prog_name,
        node=leaf_prog_name,
        sources=_pick_sources(prog.program_urls, pres.sources_edd),
        additional_instruction=(
            "Confirm that the program's official name corresponds to 'Executive Doctorate in Higher Education Management' (EHEM or similar). "
            "Allow reasonable naming variants (e.g., 'Executive Doctorate in Higher Education Management (Ed.D.)')."
        )
    )

    # Leaf: Program Institution and School
    leaf_prog_inst_school = evaluator.add_leaf(
        id="program_institution_school",
        desc="Identify the university and the specific graduate school of education that offers this Executive Doctorate in Higher Education Management program.",
        parent=program_node,
        critical=True
    )
    prog_inst = prog.institution or pres.edd_institution or "the relevant university"
    prog_school = prog.school or pres.edd_school or "the Graduate School of Education"
    claim_prog_inst_school = (
        f"The cited sources indicate this Executive Doctorate in Higher Education Management program is offered by the {prog_school} at {prog_inst}."
    )
    await evaluator.verify(
        claim=claim_prog_inst_school,
        node=leaf_prog_inst_school,
        sources=_pick_sources(prog.program_urls, pres.sources_edd),
        additional_instruction=(
            "Verify that the program belongs to a Graduate School of Education (or equivalent) at the specified institution."
        )
    )

    # ---------------------------------------------------------
    # 3) Director Details (Parallel, Critical)
    # ---------------------------------------------------------
    director_node = evaluator.add_parallel(
        id="director_details",
        desc="Provide the requested information about the program’s current director as of the 2025–2026 academic year.",
        parent=task_node,
        critical=True
    )

    # Leaf: Director Full Name
    leaf_dir_name = evaluator.add_leaf(
        id="director_full_name",
        desc="Provide the full name of the current program director (as of the 2025–2026 academic year).",
        parent=director_node,
        critical=True
    )
    director_name = prog.director_full_name or "the program's director"
    claim_dir_name = (
        f"The cited sources show that the director of the Executive Doctorate in Higher Education Management program "
        f"(for the 2025–2026 academic year) is {director_name}."
    )
    await evaluator.verify(
        claim=claim_dir_name,
        node=leaf_dir_name,
        sources=_pick_sources(prog.director_urls, prog.program_urls, pres.sources_edd),
        additional_instruction=(
            "Check program pages, announcements, or faculty profiles indicating who serves as Program Director for the Executive Doctorate in Higher Education Management "
            "as of the 2025–2026 academic year. If the page indicates a 2025 assumption and 'current Program Director', that suffices."
        )
    )

    # Leaf: Director Assumption Timing (in 2025)
    leaf_dir_assume = evaluator.add_leaf(
        id="director_assumption_timing_2025",
        desc="State when in 2025 the director assumed the director role (date or at least month/year, as supported by sources).",
        parent=director_node,
        critical=True
    )
    director_assumed = prog.director_assumption_date_2025 or "a specific month in 2025"
    claim_dir_assume = (
        f"The cited sources indicate that {director_name} assumed the director role in {director_assumed} (in 2025)."
    )
    await evaluator.verify(
        claim=claim_dir_assume,
        node=leaf_dir_assume,
        sources=_pick_sources(prog.director_urls, prog.program_urls, pres.sources_edd),
        additional_instruction=(
            "Verify the assumption timing in 2025; month and year are sufficient if a specific day is not provided. "
            "Phrasings like 'effective July 2025' or 'beginning in September 2025' are acceptable."
        )
    )

    # Leaf: Director Prior Position or Affiliation
    leaf_dir_prior = evaluator.add_leaf(
        id="director_prior_position",
        desc="Provide one prior professional position or affiliation held by the director before taking on the directorship.",
        parent=director_node,
        critical=True
    )
    prior_pos = prog.director_prior_position or "a prior professional position or affiliation"
    claim_dir_prior = (
        f"The cited sources indicate that before becoming director, {director_name} held the position/affiliation: {prior_pos}."
    )
    await evaluator.verify(
        claim=claim_dir_prior,
        node=leaf_dir_prior,
        sources=_pick_sources(prog.director_urls, prog.program_urls),
        additional_instruction=(
            "Confirm that the stated prior role or affiliation is explicitly mentioned on the cited page(s). "
            "Reasonable paraphrasing is acceptable as long as the position/affiliation is clearly supported."
        )
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the EdD program director identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # The overall task flows logically in stages
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

    # Parallelize extractions
    pres_task = evaluator.extract(
        prompt=prompt_extract_president_anchor(),
        template_class=PresidentAnchorExtraction,
        extraction_name="president_anchor"
    )
    prog_task = evaluator.extract(
        prompt=prompt_extract_program_director(),
        template_class=ProgramDirectorExtraction,
        extraction_name="program_director_details"
    )
    pres_extracted, prog_extracted = await asyncio.gather(pres_task, prog_task)

    # Add custom info for debugging/context
    evaluator.add_custom_info(
        {
            "president_name": pres_extracted.president_name,
            "university": pres_extracted.university,
            "appointment_date": pres_extracted.appointment_date,
            "edd_program_name": pres_extracted.edd_program_name,
            "edd_institution": pres_extracted.edd_institution,
            "program_name": prog_extracted.program_name,
            "director_full_name": prog_extracted.director_full_name,
            "director_assumed": prog_extracted.director_assumption_date_2025
        },
        info_type="extracted_summary",
        info_name="extracted_key_fields"
    )

    # Build and run verification tree
    await build_verification_tree(
        evaluator=evaluator,
        root_node=root,
        pres=pres_extracted,
        prog=prog_extracted
    )

    # Return evaluation summary
    return evaluator.get_summary()