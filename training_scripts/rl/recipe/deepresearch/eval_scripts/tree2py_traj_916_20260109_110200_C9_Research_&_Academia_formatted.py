import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cs_phd_us_top20_ai_ml_funding_apps_structure"
TASK_DESCRIPTION = (
    "Identify 4 Computer Science PhD programs in the United States that satisfy ALL of the following requirements:\n"
    "Location & Ranking:\n"
    "- The program must be located in the United States\n"
    "- The program must be ranked in the top 20 for Computer Science PhD programs according to U.S. News 2024 rankings\n\n"
    "Research Focus:\n"
    "- The program must have active research groups specifically in Artificial Intelligence or Machine Learning\n"
    "- The Computer Science department must have at least 50 tenure-track faculty members\n"
    "- The program must have at least 3 distinct research labs or groups focused on AI/ML subdisciplines (such as computer vision, natural language processing, robotics, deep learning, etc.)\n"
    "- At least 5 faculty members must be actively publishing in top-tier AI/ML conferences (NeurIPS, ICML, CVPR, ICLR, or AAAI)\n\n"
    "Funding:\n"
    "- The program must offer PhD stipends of at least $35,000 per year (on a 12-month basis)\n"
    "- The program must provide full funding guarantees (covering both tuition and stipend) for admitted PhD students\n\n"
    "Application Requirements:\n"
    "- The program must NOT require GRE scores for PhD admission (GRE must be optional or not accepted)\n"
    "- The PhD application deadline for Fall 2026 admission must fall between December 1, 2025, and January 15, 2026\n"
    "- The program must state a minimum or recommended GPA of at least 3.0 on a 4.0 scale\n"
    "- For international applicants, the program must accept TOEFL iBT scores with a stated minimum score requirement\n\n"
    "Program Structure:\n"
    "- PhD students must be required to serve as Teaching Assistants (TAs) for at least 1 semester during their program\n"
    "- The program must admit at least 20 PhD students per year in Computer Science\n"
    "- The average PhD completion time must be between 5 and 7 years\n\n"
    "For each of the 4 programs you identify, provide:\n"
    "1. The full official name of the university\n"
    "2. The official name of the Computer Science department or school\n"
    "3. The city and state location\n"
    "4. Verification that all requirements are met with specific details\n"
    "5. Supporting URL references for each major category of information (basic program info, research capabilities, funding, application requirements, and program structure)"
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramEntry(BaseModel):
    # Basic info
    university_name: Optional[str] = None
    department_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    # URL sets for verification (each category should have at least one URL)
    basic_info_urls: List[str] = Field(default_factory=list)
    ranking_urls: List[str] = Field(default_factory=list)  # Prefer U.S. News 2024; if not available, any official page that states rank
    research_urls: List[str] = Field(default_factory=list)
    funding_urls: List[str] = Field(default_factory=list)
    application_urls: List[str] = Field(default_factory=list)
    structure_urls: List[str] = Field(default_factory=list)

    # Optional supportive lists (not strictly required for verification, but helpful context)
    ai_ml_lab_names: List[str] = Field(default_factory=list)
    top_conf_faculty_names: List[str] = Field(default_factory=list)


class ProgramsExtraction(BaseModel):
    programs: List[ProgramEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return (
        "Extract up to 4 Computer Science PhD programs described in the answer. If the answer includes more than 4, "
        "extract only the first 4 in order of appearance. For each program, extract the following fields:\n"
        "- university_name: The full official university name.\n"
        "- department_name: The official name of the Computer Science department or school.\n"
        "- city: The city where the program is located.\n"
        "- state: The U.S. state where the program is located (use two-letter abbreviation if possible; otherwise the full state name).\n"
        "- basic_info_urls: A list of URLs that provide general/basic information about the program (official university or department pages that show name and location).\n"
        "- ranking_urls: A list of URLs supporting that the program is top-20 in U.S. News 2024 Computer Science PhD rankings. Prefer U.S. News pages or official pages explicitly citing the 2024 ranking. If none are present, leave empty.\n"
        "- research_urls: A list of URLs supporting research capabilities (AI/ML groups, labs, faculty counts, publication activity).\n"
        "- funding_urls: A list of URLs supporting funding details (stipend amount, 12-month basis, full funding guarantee).\n"
        "- application_urls: A list of URLs supporting application requirements (GRE policy, Fall 2026 deadline, minimum GPA, TOEFL iBT acceptance and minimum scores).\n"
        "- structure_urls: A list of URLs supporting program structure (TA requirement, admits per year, average time to completion).\n"
        "- ai_ml_lab_names: Names of distinct AI/ML-oriented labs or groups mentioned (up to 6). Leave empty if not explicitly named.\n"
        "- top_conf_faculty_names: Names of faculty who publish in top-tier AI/ML venues (NeurIPS/ICML/CVPR/ICLR/AAAI), if explicitly listed (up to 10). Leave empty if not provided.\n\n"
        "Apply the SPECIAL RULES FOR URL SOURCES EXTRACTION strictly. Only extract URLs explicitly present in the answer text. "
        "If a category has no explicit URLs in the answer, return an empty list for that category.\n"
        "Do not invent or infer any information. When a field is not present, set it to null (for strings) or empty list (for URL lists).\n"
        "Return a JSON object with a single key 'programs' containing an array of up to 4 ProgramEntry objects."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _urls_or_empty(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Filter blatantly invalid entries (very light check)
    cleaned = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    return cleaned


def _ordinal(i: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(i, f"Program {i}")


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    parent_node,
    program: ProgramEntry,
    index1_based: int,
) -> None:
    """
    Build and execute all verification nodes for one program.
    The node IDs and descriptions follow the rubric JSON.
    """
    # Program container node (non-critical, parallel)
    program_node = evaluator.add_parallel(
        id=f"program_{index1_based}",
        desc=f"{_ordinal(index1_based)} qualifying PhD program",
        parent=parent_node,
        critical=False,
    )

    # ------------------------ BASIC INFO ------------------------
    basic_info_node = evaluator.add_parallel(
        id=f"basic_info_{index1_based}",
        desc=f"Basic program information for Program {index1_based}",
        parent=program_node,
        critical=True,  # All children under this must be critical (framework constraint)
    )

    # URL presence for basic info (critical)
    basic_info_urls = _urls_or_empty(program.basic_info_urls)
    bi_url_presence_node = evaluator.add_custom_node(
        result=len(basic_info_urls) > 0,
        id=f"basic_info_url_{index1_based}",
        desc=f"Provide URL reference for basic program information",
        parent=basic_info_node,
        critical=True,
    )

    # University name verification (critical)
    uni_node = evaluator.add_leaf(
        id=f"university_name_{index1_based}",
        desc="Provide the full official name of the university",
        parent=basic_info_node,
        critical=True,
    )
    uni_claim = f"The full official name of the university is '{program.university_name}'."
    await evaluator.verify(
        claim=uni_claim,
        node=uni_node,
        sources=basic_info_urls,
        extra_prerequisites=[bi_url_presence_node],
        additional_instruction="Check the official site or department page to confirm the exact university name. Allow minor punctuation or capitalization differences.",
    )

    # Department name verification (critical)
    dept_node = evaluator.add_leaf(
        id=f"department_name_{index1_based}",
        desc="Provide the official name of the Computer Science department or school",
        parent=basic_info_node,
        critical=True,
    )
    dept_claim = (
        f"The official name of the Computer Science department or school is '{program.department_name}'."
    )
    await evaluator.verify(
        claim=dept_claim,
        node=dept_node,
        sources=basic_info_urls,
        extra_prerequisites=[bi_url_presence_node],
        additional_instruction="Verify the department or school name on the official pages. Allow minor variants that clearly refer to the same official entity.",
    )

    # Location and US verification (critical)
    loc_node = evaluator.add_leaf(
        id=f"location_us_{index1_based}",
        desc="Verify the program is located in the United States with city and state",
        parent=basic_info_node,
        critical=True,
    )
    loc_city = program.city if program.city else ""
    loc_state = program.state if program.state else ""
    loc_claim = (
        f"The program is located in the United States, specifically in {loc_city}, {loc_state}."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=basic_info_urls,
        extra_prerequisites=[bi_url_presence_node],
        additional_instruction="Confirm that the institution is in the U.S. and that the stated city and state are correct.",
    )

    # Ranking top-20 in U.S. News 2024 (critical)
    rank_node = evaluator.add_leaf(
        id=f"ranking_verification_{index1_based}",
        desc="Verify the program ranks in the top 20 for CS PhD according to U.S. News 2024 rankings",
        parent=basic_info_node,
        critical=True,
    )
    ranking_sources = _urls_or_empty(program.ranking_urls) or basic_info_urls
    rank_claim = (
        f"The Computer Science PhD program at {program.university_name} is ranked within the top 20 "
        f"in the U.S. News 2024 Computer Science rankings."
    )
    await evaluator.verify(
        claim=rank_claim,
        node=rank_node,
        sources=ranking_sources,
        extra_prerequisites=[bi_url_presence_node],
        additional_instruction="Confirm the U.S. News & World Report 2024 Computer Science (PhD) ranking is top-20. If the page is unrelated or does not show 2024 CS ranking, mark as not supported.",
    )

    # ------------------------ RESEARCH CAPABILITIES ------------------------
    research_node = evaluator.add_parallel(
        id=f"research_capabilities_{index1_based}",
        desc=f"Research capabilities and focus areas for Program {index1_based}",
        parent=program_node,
        critical=True,
    )
    research_urls = _urls_or_empty(program.research_urls)
    research_url_presence = evaluator.add_custom_node(
        result=len(research_urls) > 0,
        id=f"research_url_{index1_based}",
        desc="Provide URL reference for research information",
        parent=research_node,
        critical=True,
    )

    # AI/ML research groups exist
    aiml_node = evaluator.add_leaf(
        id=f"ai_ml_research_{index1_based}",
        desc="Verify active research groups in Artificial Intelligence or Machine Learning exist",
        parent=research_node,
        critical=True,
    )
    aiml_claim = "The CS department has active research groups in Artificial Intelligence or Machine Learning."
    await evaluator.verify(
        claim=aiml_claim,
        node=aiml_node,
        sources=research_urls,
        extra_prerequisites=[research_url_presence],
        additional_instruction="Look for research area pages, lab pages, or group pages that explicitly reference AI/ML activities.",
    )

    # Faculty size >= 50 tenure-track
    faculty_node = evaluator.add_leaf(
        id=f"faculty_size_{index1_based}",
        desc="Verify at least 50 tenure-track faculty members in CS department",
        parent=research_node,
        critical=True,
    )
    faculty_claim = "The Computer Science department has at least 50 tenure-track faculty members."
    await evaluator.verify(
        claim=faculty_claim,
        node=faculty_node,
        sources=research_urls,
        extra_prerequisites=[research_url_presence],
        additional_instruction="Look for a faculty count or list that specifically refers to tenure-track or tenured/tenure-track; if clear count ≥ 50 is not supported, mark as not supported.",
    )

    # At least 3 distinct AI/ML labs/groups
    labs_node = evaluator.add_leaf(
        id=f"research_labs_{index1_based}",
        desc="Verify at least 3 distinct AI/ML research labs or groups",
        parent=research_node,
        critical=True,
    )
    sample_labs = ", ".join(program.ai_ml_lab_names[:3]) if program.ai_ml_lab_names else ""
    labs_claim = (
        "There are at least 3 distinct AI/ML research labs or groups in the CS department."
        + (f" Examples include: {sample_labs}." if sample_labs else "")
    )
    await evaluator.verify(
        claim=labs_claim,
        node=labs_node,
        sources=research_urls,
        extra_prerequisites=[research_url_presence],
        additional_instruction="Confirm there are three or more separate labs/groups explicitly focused on AI/ML subareas such as NLP, CV, robotics, or deep learning.",
    )

    # At least 5 faculty active in top-tier AI/ML venues
    topconf_node = evaluator.add_leaf(
        id=f"top_conference_faculty_{index1_based}",
        desc="Verify at least 5 faculty actively publishing in top-tier AI/ML conferences (NeurIPS, ICML, CVPR, ICLR, AAAI)",
        parent=research_node,
        critical=True,
    )
    sample_faculty = ", ".join(program.top_conf_faculty_names[:5]) if program.top_conf_faculty_names else ""
    topconf_claim = (
        "At least 5 faculty members are actively publishing in top-tier AI/ML conferences such as NeurIPS, ICML, CVPR, ICLR, or AAAI."
        + (f" Examples include: {sample_faculty}." if sample_faculty else "")
    )
    await evaluator.verify(
        claim=topconf_claim,
        node=topconf_node,
        sources=research_urls,
        extra_prerequisites=[research_url_presence],
        additional_instruction="Confirm evidence that five or more faculty are active in these venues; acceptable evidence includes publication lists or faculty CVs linked from official pages.",
    )

    # ------------------------ FUNDING INFO ------------------------
    funding_node = evaluator.add_parallel(
        id=f"funding_info_{index1_based}",
        desc=f"Funding and financial support for Program {index1_based}",
        parent=program_node,
        critical=True,
    )
    funding_urls = _urls_or_empty(program.funding_urls)
    funding_url_presence = evaluator.add_custom_node(
        result=len(funding_urls) > 0,
        id=f"funding_url_{index1_based}",
        desc="Provide URL reference for funding information",
        parent=funding_node,
        critical=True,
    )

    # Stipend >= $35k on 12-month basis
    stipend_node = evaluator.add_leaf(
        id=f"stipend_amount_{index1_based}",
        desc="Verify PhD stipend is at least $35,000 per year (12-month basis)",
        parent=funding_node,
        critical=True,
    )
    stipend_claim = "The PhD stipend is at least $35,000 per year on a 12-month basis."
    await evaluator.verify(
        claim=stipend_claim,
        node=stipend_node,
        sources=funding_urls,
        extra_prerequisites=[funding_url_presence],
        additional_instruction="Confirm the stipend figure and that it is for a 12-month period; if unclear or a 9-month figure is stated without 12-month equivalence, mark as not supported.",
    )

    # Full funding guarantee
    fullfund_node = evaluator.add_leaf(
        id=f"full_funding_{index1_based}",
        desc="Verify full funding guarantee (tuition + stipend) for admitted PhD students",
        parent=funding_node,
        critical=True,
    )
    fullfund_claim = "Admitted PhD students receive full funding that covers tuition and provides a stipend."
    await evaluator.verify(
        claim=fullfund_claim,
        node=fullfund_node,
        sources=funding_urls,
        extra_prerequisites=[funding_url_presence],
        additional_instruction="Look for explicit statements guaranteeing full funding (tuition and stipend) for all admitted PhD students.",
    )

    # ------------------------ APPLICATION REQUIREMENTS ------------------------
    apps_node = evaluator.add_parallel(
        id=f"application_requirements_{index1_based}",
        desc=f"Application requirements and deadlines for Program {index1_based}",
        parent=program_node,
        critical=True,
    )
    application_urls = _urls_or_empty(program.application_urls)
    application_url_presence = evaluator.add_custom_node(
        result=len(application_urls) > 0,
        id=f"application_url_{index1_based}",
        desc="Provide URL reference for application requirements",
        parent=apps_node,
        critical=True,
    )

    # GRE not required
    gre_node = evaluator.add_leaf(
        id=f"gre_requirement_{index1_based}",
        desc="Verify GRE is not required (optional or not accepted) for PhD admission",
        parent=apps_node,
        critical=True,
    )
    gre_claim = "The GRE is not required for PhD admission (it is optional or not accepted)."
    await evaluator.verify(
        claim=gre_claim,
        node=gre_node,
        sources=application_urls,
        extra_prerequisites=[application_url_presence],
        additional_instruction="Confirm the policy explicitly states GRE not required; if it is required, mark as not supported.",
    )

    # Deadline range for Fall 2026: between Dec 1, 2025 and Jan 15, 2026 inclusive
    deadline_node = evaluator.add_leaf(
        id=f"application_deadline_{index1_based}",
        desc="Verify PhD application deadline falls between December 1 and January 15 for Fall 2026",
        parent=apps_node,
        critical=True,
    )
    deadline_claim = (
        "The PhD application deadline for Fall 2026 admission falls between December 1, 2025 and January 15, 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_node,
        sources=application_urls,
        extra_prerequisites=[application_url_presence],
        additional_instruction="Identify the Fall 2026 PhD deadline; confirm it lies within the inclusive range Dec 1, 2025 to Jan 15, 2026. If only another term/year is shown, mark as not supported.",
    )

    # Minimum GPA >= 3.0 on 4.0 scale
    gpa_node = evaluator.add_leaf(
        id=f"minimum_gpa_{index1_based}",
        desc="Verify stated minimum or recommended GPA of at least 3.0 on 4.0 scale",
        parent=apps_node,
        critical=True,
    )
    gpa_claim = "The program states a minimum or recommended GPA of at least 3.0 on a 4.0 scale."
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_node,
        sources=application_urls,
        extra_prerequisites=[application_url_presence],
        additional_instruction="Look for explicit GPA thresholds; if unclear or below 3.0, mark as not supported.",
    )

    # TOEFL iBT acceptance with stated minimum
    toefl_node = evaluator.add_leaf(
        id=f"english_proficiency_{index1_based}",
        desc="For international applicants, verify TOEFL iBT acceptance with stated minimum score",
        parent=apps_node,
        critical=True,
    )
    toefl_claim = "The program accepts TOEFL iBT and states a minimum score requirement."
    await evaluator.verify(
        claim=toefl_claim,
        node=toefl_node,
        sources=application_urls,
        extra_prerequisites=[application_url_presence],
        additional_instruction="Confirm TOEFL iBT is accepted and that a numeric minimum is specified (overall or component).",
    )

    # ------------------------ PROGRAM STRUCTURE ------------------------
    structure_node = evaluator.add_parallel(
        id=f"program_structure_{index1_based}",
        desc=f"Program structure and requirements for Program {index1_based}",
        parent=program_node,
        critical=True,
    )
    structure_urls = _urls_or_empty(program.structure_urls)
    structure_url_presence = evaluator.add_custom_node(
        result=len(structure_urls) > 0,
        id=f"structure_url_{index1_based}",
        desc="Provide URL reference for program structure information",
        parent=structure_node,
        critical=True,
    )

    # TA requirement at least 1 semester
    ta_node = evaluator.add_leaf(
        id=f"ta_requirement_{index1_based}",
        desc="Verify PhD students must serve as TAs for at least 1 semester",
        parent=structure_node,
        critical=True,
    )
    ta_claim = "PhD students are required to serve as Teaching Assistants for at least one semester during the program."
    await evaluator.verify(
        claim=ta_claim,
        node=ta_node,
        sources=structure_urls,
        extra_prerequisites=[structure_url_presence],
        additional_instruction="Look for degree requirements stating mandatory TA service for at least one term.",
    )

    # Program admits >= 20 per year
    size_node = evaluator.add_leaf(
        id=f"program_size_{index1_based}",
        desc="Verify program admits at least 20 PhD students per year in CS",
        parent=structure_node,
        critical=True,
    )
    size_claim = "The Computer Science PhD program admits at least 20 students per year."
    await evaluator.verify(
        claim=size_claim,
        node=size_node,
        sources=structure_urls,
        extra_prerequisites=[structure_url_presence],
        additional_instruction="Confirm cohort size or annual admissions count is 20 or more.",
    )

    # Average completion time 5-7 years
    time_node = evaluator.add_leaf(
        id=f"completion_time_{index1_based}",
        desc="Verify average PhD completion time is 5-7 years",
        parent=structure_node,
        critical=True,
    )
    time_claim = "The average time to complete the PhD is between 5 and 7 years."
    await evaluator.verify(
        claim=time_claim,
        node=time_node,
        sources=structure_urls,
        extra_prerequisites=[structure_url_presence],
        additional_instruction="Check official handbooks or FAQ pages for average completion time; 5–7 years inclusive.",
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
    Entry point for evaluating an answer to the CS PhD program selection task.
    """
    # Initialize evaluator (root is always non-critical by framework design)
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

    # Extract structured information about up to 4 programs
    extracted: ProgramsExtraction = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction",
    )

    # Normalize to exactly 4 entries (pad with empty objects if fewer)
    programs: List[ProgramEntry] = list(extracted.programs[:4])
    while len(programs) < 4:
        programs.append(ProgramEntry())

    # Build verification tree according to rubric for 4 programs
    # The top-level rubric root in JSON is critical, but the framework's root is non-critical.
    # We enforce strictness by making each program's 5 categories critical; each category has only critical leaves.
    for idx in range(4):
        await verify_program(
            evaluator=evaluator,
            parent_node=root,
            program=programs[idx],
            index1_based=idx + 1,
        )

    # Return the final evaluation summary
    return evaluator.get_summary()