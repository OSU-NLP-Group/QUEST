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
TASK_ID = "financial_advisor_programs"
TASK_DESCRIPTION = """
I am considering a career transition into financial advisory and want to identify structured training programs offered by major firms. Please identify three financial advisor development or training programs from major U.S. financial services or wealth management firms that meet ALL of the following criteria:

1. The program must be a formal, named financial advisor development/training program (not just a general job posting)
2. The program must have a defined structured training period of at least 12 months
3. The program must explicitly provide support, training, or sponsorship for obtaining at least one FINRA securities license (Series 7, Series 65, Series 66, or SIE)
4. The program must offer a base salary or guaranteed compensation during the training period (not purely commission-based from the start)
5. The firm must have physical office locations or branches in at least one of these Midwest states: Ohio, Indiana, Illinois, Michigan, or Wisconsin
6. The program's minimum education requirement must be clearly stated
7. There must be a publicly accessible way to learn about or apply to the program (through an official website, career page, or program portal)
8. The program must include structured training components beyond just licensing exam preparation (such as client relationship skills, financial planning education, mentorship, or business development training)

For each of the three programs, provide: the official program name, the firm offering the program, the training duration, the specific FINRA license(s) supported, evidence of base salary or guaranteed compensation during training, at least one specific Midwest state where the firm has offices, the minimum education requirement, how to access information about or apply to the program, the specific professional development components included beyond licensing exam prep, and direct URLs to official sources supporting this information.
"""

ALLOWED_MIDWEST_STATES = {"Ohio", "Indiana", "Illinois", "Michigan", "Wisconsin"}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramInfo(BaseModel):
    program_name: Optional[str] = None
    firm: Optional[str] = None

    identification_url: Optional[str] = None  # Official program page URL for identification/structure
    training_duration_text: Optional[str] = None  # e.g., "18 months", "12-24 months", "two years"

    license_types_supported: List[str] = Field(default_factory=list)  # e.g., ["Series 7", "SIE"]
    license_support_nature: Optional[str] = None  # e.g., "sponsorship", "training", "exam fee coverage"
    licensing_compensation_urls: List[str] = Field(default_factory=list)  # URLs supporting licensing/comp claims

    base_salary_evidence_text: Optional[str] = None
    not_commission_only_evidence_text: Optional[str] = None

    midwest_states_with_offices: List[str] = Field(default_factory=list)  # state names where firm has offices
    office_evidence_urls: List[str] = Field(default_factory=list)  # URLs showing office locations/branches

    minimum_education_requirement: Optional[str] = None

    application_access_url: Optional[str] = None  # public page to learn/apply

    professional_development_components: List[str] = Field(default_factory=list)  # e.g., ["mentorship", "business development"]
    location_access_development_urls: List[str] = Field(default_factory=list)  # URLs supporting location/education/dev info


class ProgramsExtraction(BaseModel):
    programs: List[ProgramInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract up to three financial advisor development/training programs mentioned in the answer. Focus on formal, named programs (not generic job postings).
    For each program, extract the following fields exactly as stated in the answer:

    1. program_name: The official program name (string).
    2. firm: The major financial services/wealth management firm offering the program (string).
    3. identification_url: A direct URL to an official page describing the program (string URL). If missing, return null.
    4. training_duration_text: The stated structured training duration (string, e.g., "18 months", "12-24 months"). If missing, return null.
    5. license_types_supported: A list of specific FINRA license names explicitly mentioned as supported (e.g., "Series 7", "Series 65", "Series 66", "SIE"). If none are mentioned, return an empty list.
    6. license_support_nature: The nature of support provided for licensing (string, e.g., "sponsorship", "training", "exam fee coverage"). If not mentioned, return null.
    7. licensing_compensation_urls: A list of official URLs that support licensing and/or compensation information. If none are present, return an empty list.
    8. base_salary_evidence_text: Text phrasing that indicates base salary or guaranteed compensation during training (string). If not mentioned, return null.
    9. not_commission_only_evidence_text: Text indicating compensation is not purely commission-based from the start (string). If not mentioned, return null.
    10. midwest_states_with_offices: A list of specific Midwest states (Ohio, Indiana, Illinois, Michigan, Wisconsin) explicitly mentioned in the answer where the firm has offices/branches. If none are mentioned, return an empty list.
    11. office_evidence_urls: A list of official URLs showing physical office locations/branches (e.g., branch locator or office listings). If none are present, return an empty list.
    12. minimum_education_requirement: The clearly stated minimum education requirement for the program (string). If not mentioned, return null.
    13. application_access_url: A publicly accessible URL to learn about or apply to the program (string URL). If missing, return null.
    14. professional_development_components: A list of structured training components beyond licensing exam prep (e.g., "client relationship skills", "financial planning education", "mentorship", "business development"). If not mentioned, return an empty list.
    15. location_access_development_urls: A list of official URLs that support geographic presence, education requirement, application access, or professional development components. If none are present, return an empty list.

    IMPORTANT:
    - Extract only URLs that are explicitly present in the answer. Do not invent or infer URLs.
    - Include full URLs with protocol. If the answer shows a URL without protocol, prepend "http://".
    - If the answer contains more than three programs, extract only the first three mentioned.
    - If any field is missing for a program, set it to null (or empty list where appropriate).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*sources_groups: List[Optional[str] | List[str] | None]) -> List[str]:
    urls: List[str] = []
    for group in sources_groups:
        if group is None:
            continue
        if isinstance(group, list):
            for u in group:
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())
        elif isinstance(group, str):
            if group.strip():
                urls.append(group.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _pick_allowed_midwest_state(states: List[str]) -> Optional[str]:
    for s in states:
        if s:
            name = s.strip()
            # Normalize capitalization and compare
            if name.title() in ALLOWED_MIDWEST_STATES:
                return name.title()
    return None


# --------------------------------------------------------------------------- #
# Verification logic per program                                              #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    parent_node,
    program: ProgramInfo,
    program_index: int,
) -> None:
    """
    Build verification nodes and run checks for a single program.
    """
    # Program node (non-critical to allow partial across programs)
    pg_node = evaluator.add_parallel(
        id=f"program_{program_index+1}",
        desc=f"{['First','Second','Third'][program_index]} financial advisor development program meeting all criteria",
        parent=parent_node,
        critical=False
    )

    # ---------------- Identification & Structure (Critical) ---------------- #
    ident_node = evaluator.add_parallel(
        id=f"program_{program_index+1}_identification_and_structure",
        desc="Program identification and training structure requirements are met",
        parent=pg_node,
        critical=True
    )

    # Program details (Critical)
    details_node = evaluator.add_parallel(
        id=f"program_{program_index+1}_program_details",
        desc="Official program name and firm offering it are identified",
        parent=ident_node,
        critical=True
    )

    # Leaf: Program name
    name_leaf = evaluator.add_leaf(
        id=f"program_{program_index+1}_name",
        desc="The official name of the financial advisor development/training program is stated",
        parent=details_node,
        critical=True
    )
    name_claim = f"The official program page describes a formal, named advisor development/training program called '{program.program_name}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=program.identification_url,
        additional_instruction="Verify the page uses an official program name for a formal advisor development/training program (allow minor styling/branding variations)."
    )

    # Leaf: Firm
    firm_leaf = evaluator.add_leaf(
        id=f"program_{program_index+1}_firm",
        desc="The major financial services firm offering the program is identified",
        parent=details_node,
        critical=True
    )
    firm_claim = f"The program is offered by {program.firm}."
    await evaluator.verify(
        claim=firm_claim,
        node=firm_leaf,
        sources=program.identification_url,
        additional_instruction="Confirm the offering firm is clearly indicated on the official program page."
    )

    # Duration requirement (Critical)
    duration_node = evaluator.add_parallel(
        id=f"program_{program_index+1}_duration_requirement",
        desc="Training duration meets minimum requirement",
        parent=ident_node,
        critical=True
    )

    # Leaf: Duration stated
    duration_stated_leaf = evaluator.add_leaf(
        id=f"program_{program_index+1}_duration_stated",
        desc="The program's structured training period duration is clearly stated",
        parent=duration_node,
        critical=True
    )
    duration_stated_claim = f"The program page clearly states the structured training duration as '{program.training_duration_text}'."
    await evaluator.verify(
        claim=duration_stated_claim,
        node=duration_stated_leaf,
        sources=program.identification_url,
        additional_instruction="Confirm the page explicitly mentions a structured training period and its duration (e.g., months/years)."
    )

    # Leaf: Duration meets minimum 12 months
    duration_min_leaf = evaluator.add_leaf(
        id=f"program_{program_index+1}_duration_meets_minimum",
        desc="The stated duration is at least 12 months",
        parent=duration_node,
        critical=True
    )
    duration_min_claim = "The program has a structured training period of at least 12 months."
    await evaluator.verify(
        claim=duration_min_claim,
        node=duration_min_leaf,
        sources=program.identification_url,
        additional_instruction="Check the page for a duration of 12 months or longer (e.g., 18 months, 24 months, or '12-24 months')."
    )

    # Leaf: Identification URL existence (Critical)
    ident_url_exists = evaluator.add_custom_node(
        result=(program.identification_url is not None and program.identification_url.strip() != ""),
        id=f"program_{program_index+1}_identification_url",
        desc="URL to official program page or description is provided",
        parent=ident_node,
        critical=True
    )

    # --------------- Licensing & Compensation (Critical) ------------------ #
    lic_comp_node = evaluator.add_parallel(
        id=f"program_{program_index+1}_licensing_and_compensation",
        desc="Licensing support and compensation structure requirements are met",
        parent=pg_node,
        critical=True
    )

    # License provision (Critical)
    lic_prov_node = evaluator.add_parallel(
        id=f"program_{program_index+1}_license_provision",
        desc="Program provides support for securities licensing",
        parent=lic_comp_node,
        critical=True
    )

    # Leaf: License type mentioned
    license_type_leaf = evaluator.add_leaf(
        id=f"program_{program_index+1}_license_type",
        desc="At least one specific FINRA license (Series 7, 65, 66, or SIE) is mentioned as supported by the program",
        parent=lic_prov_node,
        critical=True
    )
    if program.license_types_supported:
        lt_display = ", ".join(program.license_types_supported)
        license_type_claim = f"The program explicitly mentions support for obtaining at least one FINRA license: {lt_display}."
    else:
        license_type_claim = "The official page explicitly mentions support for obtaining at least one FINRA license among Series 7, Series 65, Series 66, or SIE."
    await evaluator.verify(
        claim=license_type_claim,
        node=license_type_leaf,
        sources=_merge_sources(program.licensing_compensation_urls, program.identification_url),
        additional_instruction="Look for explicit mentions of Series 7, Series 65, Series 66, or SIE. Synonyms like 'Series-7' or 'Securities Industry Essentials (SIE)' should count."
    )

    # Leaf: Nature of licensing support
    license_support_leaf = evaluator.add_leaf(
        id=f"program_{program_index+1}_license_support_nature",
        desc="The nature of support (training, sponsorship, or exam fee coverage) is described",
        parent=lic_prov_node,
        critical=True
    )
    if program.license_support_nature and program.license_support_nature.strip():
        ls_claim = f"The program provides {program.license_support_nature.strip()} for licensing."
    else:
        ls_claim = "The program provides licensing support such as training, sponsorship, or exam fee coverage."
    await evaluator.verify(
        claim=ls_claim,
        node=license_support_leaf,
        sources=_merge_sources(program.licensing_compensation_urls, program.identification_url),
        additional_instruction="Confirm that the page describes the nature of licensing support (e.g., study materials, paid sponsorship, exam fee coverage, structured training)."
    )

    # Compensation details (Critical)
    comp_node = evaluator.add_parallel(
        id=f"program_{program_index+1}_compensation_details",
        desc="Program compensation structure meets requirements",
        parent=lic_comp_node,
        critical=True
    )

    # Leaf: Base salary or guaranteed comp
    base_salary_leaf = evaluator.add_leaf(
        id=f"program_{program_index+1}_base_salary",
        desc="Evidence shows the program offers base salary or guaranteed compensation during training",
        parent=comp_node,
        critical=True
    )
    base_salary_claim = "The program offers a base salary or guaranteed compensation during the training period."
    await evaluator.verify(
        claim=base_salary_claim,
        node=base_salary_leaf,
        sources=_merge_sources(program.licensing_compensation_urls, program.identification_url),
        additional_instruction="Accept phrasing such as 'base pay', 'salary', 'guaranteed compensation', 'stipend', or 'salary plus bonus' during training."
    )

    # Leaf: Not commission-only from start
    not_commission_leaf = evaluator.add_leaf(
        id=f"program_{program_index+1}_not_commission_only",
        desc="Evidence confirms compensation is not purely commission-based from the start",
        parent=comp_node,
        critical=True
    )
    not_commission_claim = "Compensation is not purely commission-based from the start; trainees receive salary/guaranteed pay."
    await evaluator.verify(
        claim=not_commission_claim,
        node=not_commission_leaf,
        sources=_merge_sources(program.licensing_compensation_urls, program.identification_url),
        additional_instruction="Verify that trainees are not 100% commission at the outset; evidence of base/guaranteed pay suffices."
    )

    # Leaf: Licensing/compensation URL existence (Critical)
    lic_comp_url_exists = evaluator.add_custom_node(
        result=(bool(program.licensing_compensation_urls) and len(program.licensing_compensation_urls) > 0),
        id=f"program_{program_index+1}_licensing_compensation_url",
        desc="URL reference supporting licensing and compensation information is provided",
        parent=lic_comp_node,
        critical=True
    )

    # -------- Location, Access, Education, Professional Development ------- #
    loc_acc_dev_node = evaluator.add_parallel(
        id=f"program_{program_index+1}_location_access_development",
        desc="Geographic presence, application access, education, and professional development requirements are met",
        parent=pg_node,
        critical=True
    )

    # Midwest location (Critical)
    midwest_node = evaluator.add_parallel(
        id=f"program_{program_index+1}_midwest_location",
        desc="Firm has presence in Midwest states",
        parent=loc_acc_dev_node,
        critical=True
    )

    chosen_state = _pick_allowed_midwest_state(program.midwest_states_with_offices) or ""

    # Leaf: Specific state identified
    specific_state_leaf = evaluator.add_leaf(
        id=f"program_{program_index+1}_specific_state",
        desc="At least one specific Midwest state (Ohio, Indiana, Illinois, Michigan, or Wisconsin) where the firm has offices is identified",
        parent=midwest_node,
        critical=True
    )
    specific_state_claim = f"The firm has offices/branches in {chosen_state}."
    await evaluator.verify(
        claim=specific_state_claim,
        node=specific_state_leaf,
        sources=_merge_sources(program.office_evidence_urls, program.location_access_development_urls),
        additional_instruction="Verify the firm's physical office presence in the stated Midwest state using official locator/listing pages."
    )

    # Leaf: Office evidence provided
    office_evidence_leaf = evaluator.add_leaf(
        id=f"program_{program_index+1}_office_evidence",
        desc="Evidence of physical office locations or branches in the identified state is provided",
        parent=midwest_node,
        critical=True
    )
    office_evidence_claim = f"The provided office locator or branch listing page shows physical offices/branches in {chosen_state}."
    await evaluator.verify(
        claim=office_evidence_claim,
        node=office_evidence_leaf,
        sources=program.office_evidence_urls,
        additional_instruction="Confirm the page lists addresses or branch locations in the identified state."
    )

    # Leaf: Minimum education requirement
    education_leaf = evaluator.add_leaf(
        id=f"program_{program_index+1}_education",
        desc="Minimum education requirement is clearly stated (e.g., bachelor's degree required, high school diploma, or no degree specified)",
        parent=loc_acc_dev_node,
        critical=True
    )
    education_claim = f"The program's minimum education requirement is '{program.minimum_education_requirement}'."
    await evaluator.verify(
        claim=education_claim,
        node=education_leaf,
        sources=_merge_sources(program.identification_url, program.location_access_development_urls),
        additional_instruction="Verify the page explicitly states the minimum education requirement (e.g., bachelor's degree, high school diploma)."
    )

    # Leaf: Publicly accessible application/info access
    app_access_leaf = evaluator.add_leaf(
        id=f"program_{program_index+1}_application_access",
        desc="Publicly accessible way to learn about or apply to the program is identified (official career page or program portal)",
        parent=loc_acc_dev_node,
        critical=True
    )
    app_access_claim = "This page provides information about or a way to apply to the program and is publicly accessible."
    await evaluator.verify(
        claim=app_access_claim,
        node=app_access_leaf,
        sources=program.application_access_url or program.identification_url,
        additional_instruction="Confirm the URL is an official page providing program info or application access (e.g., careers portal, program page)."
    )

    # Leaf: Professional development components beyond licensing
    prof_dev_leaf = evaluator.add_leaf(
        id=f"program_{program_index+1}_professional_development",
        desc="Structured training components beyond licensing exam prep are identified (e.g., client skills, financial planning education, mentorship, business development)",
        parent=loc_acc_dev_node,
        critical=True
    )
    if program.professional_development_components:
        pdev = ", ".join(program.professional_development_components)
        prof_dev_claim = f"The program includes structured training components beyond licensing exam prep, such as {pdev}."
    else:
        prof_dev_claim = "The program includes structured training components beyond licensing exam preparation, such as client relationship skills, financial planning education, mentorship, or business development."
    await evaluator.verify(
        claim=prof_dev_claim,
        node=prof_dev_leaf,
        sources=_merge_sources(program.identification_url, program.location_access_development_urls),
        additional_instruction="Look for explicit mentions of training elements beyond exam prep (mentorship, client skills, planning education, business development)."
    )

    # Leaf: Location/access/development URLs existence (Critical)
    loc_acc_dev_url_exists = evaluator.add_custom_node(
        result=(
            (bool(program.office_evidence_urls) and len(program.office_evidence_urls) > 0)
            or (program.application_access_url is not None and program.application_access_url.strip() != "")
            or (bool(program.location_access_development_urls) and len(program.location_access_development_urls) > 0)
        ),
        id=f"program_{program_index+1}_location_access_url",
        desc="URL reference supporting geographic presence, education, access, and development information is provided",
        parent=loc_acc_dev_node,
        critical=True
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
) -> Dict[str, Any]:
    """
    Evaluate the answer for identifying three financial advisor development programs meeting all criteria.
    """
    # Initialize evaluator (root must be non-critical due to framework constraints on critical parents)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify three financial advisor development programs from major U.S. financial services firms that meet all specified criteria",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract programs
    extraction = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction"
    )

    # Normalize to exactly 3 programs (pad with empty ProgramInfo if fewer)
    programs = list(extraction.programs[:3])
    while len(programs) < 3:
        programs.append(ProgramInfo())

    # Add a summary of constraints to ground truth info
    evaluator.add_ground_truth({
        "required_criteria": [
            "Formal, named advisor development/training program",
            "Training duration >= 12 months",
            "Support for at least one FINRA license (Series 7, 65, 66, or SIE)",
            "Base salary or guaranteed compensation during training",
            "Firm has physical offices in OH/IN/IL/MI/WI",
            "Minimum education requirement stated",
            "Publicly accessible info/apply page",
            "Professional development beyond licensing exam prep"
        ],
        "allowed_midwest_states": sorted(list(ALLOWED_MIDWEST_STATES))
    }, gt_type="criteria")

    # Build verification tree and run checks for each program
    for idx, program in enumerate(programs):
        await verify_program(evaluator, root, program, idx)

    # Return evaluation summary
    return evaluator.get_summary()