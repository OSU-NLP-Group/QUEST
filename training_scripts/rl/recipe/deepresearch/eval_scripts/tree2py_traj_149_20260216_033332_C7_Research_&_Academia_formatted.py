import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "single_us_phd_program_eval"
TASK_DESCRIPTION = """Identify one university in the United States that offers a doctoral (PhD) program in biomedical sciences, biochemistry, molecular biology, or a closely related biological/health sciences field, and provide comprehensive information about its program structure and research administration policies. The university and program must meet the following requirements:

1. The program's typical time-to-degree completion must fall within 4-7 years
2. The program must require dissertation committees consisting of at least 3 members
3. Dissertation committee members must be required to hold doctoral degrees or appropriate terminal degrees
4. The program must require that at least some dissertation committee members be full-time faculty members at the institution
5. The university must have an Institutional Review Board (IRB) for reviewing research involving human subjects
6. The university must be eligible to receive federal research grants from agencies such as the National Institutes of Health (NIH) or National Science Foundation (NSF)
7. Faculty members serving as primary dissertation advisors must be required to hold doctoral degrees in relevant fields

For the identified university and program, provide:
- The university name and the specific doctoral program name
- Documentation of the program duration (typical years to completion)
- Dissertation committee composition requirements
- Committee member qualification requirements
- IRB process information
- Evidence of federal grant eligibility
- Any additional relevant information about research security training, grant submission processes, indirect cost rates, research funding record, IRB continuing review processes, credit hour requirements, and comprehensive examination requirements

All information must be supported by official university sources, program websites, or institutional policy documents.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Evidence(BaseModel):
    """Generic evidence container for a single requirement."""
    statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ProgramIdentity(BaseModel):
    """Identity and basic classification of the program."""
    university_name: Optional[str] = None
    program_name: Optional[str] = None
    field_area: Optional[str] = None  # e.g., Biomedical Sciences, Biochemistry, Molecular Biology, or closely related
    program_page_url: Optional[str] = None
    identity_sources: List[str] = Field(default_factory=list)
    university_overview_sources: List[str] = Field(default_factory=list)


class ProgramExtraction(BaseModel):
    """Complete extraction structure covering identity and all requirements."""
    identity: Optional[ProgramIdentity] = None

    time_to_degree: Evidence = Evidence()
    committee_min_size: Evidence = Evidence()
    committee_terminal_degree: Evidence = Evidence()
    committee_full_time_requirement: Evidence = Evidence()

    irb_exists: Evidence = Evidence()
    irb_continuing_review: Evidence = Evidence()

    federal_grant_eligibility: Evidence = Evidence()
    grant_submission_system: Evidence = Evidence()
    indirect_cost_rate_info: Evidence = Evidence()
    active_federal_funding_record: Evidence = Evidence()
    research_security_training: Evidence = Evidence()

    primary_advisor_degree_requirement: Evidence = Evidence()
    credit_hours_requirement: Evidence = Evidence()
    comprehensive_exam_requirement: Evidence = Evidence()


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_program_info() -> str:
    return """
    Extract the following structured information from the answer. Only extract text explicitly present in the answer and the URLs that are explicitly provided in the answer. Do not invent or infer any new URLs.

    1) identity:
       - university_name: The name of the university.
       - program_name: The exact PhD program name.
       - field_area: The field classification (e.g., "Biomedical Sciences", "Biochemistry", "Molecular Biology", or a closely related biological/health sciences field).
       - program_page_url: A single primary official program webpage URL if provided in the answer; otherwise null.
       - identity_sources: All official URLs cited in the answer that support the program identity and field classification (including the program page).
       - university_overview_sources: Any official URLs cited in the answer that demonstrate the university overview or location (e.g., "About" page, address information).

    For the following requirements, for each item extract:
    - statement: The text or summary phrase from the answer that describes the requirement (verbatim or concise paraphrase).
    - sources: The list of official URLs cited in the answer that support that specific requirement. If multiple URLs are given in the answer for one item, include all.

    2) time_to_degree: Typical time-to-degree completion (e.g., "5-6 years", "4-7 years", "usually 5 years").
    3) committee_min_size: Minimum dissertation committee size requirement (e.g., "at least 3 members").
    4) committee_terminal_degree: Requirement that committee members hold doctoral or terminal degrees (e.g., "must hold PhD/MD or terminal degrees").
    5) committee_full_time_requirement: Requirement that at least some committee members be full-time faculty at the institution.

    6) irb_exists: Confirmation of an IRB (Institutional Review Board) process for human-subjects research.
    7) irb_continuing_review: Confirmation of an IRB continuing review/renewal process for approved studies.

    8) federal_grant_eligibility: Evidence the university is eligible to receive federal research grants (e.g., NIH/NSF eligibility, SAM/UEI registration mentions).
    9) grant_submission_system: Evidence of Grants.gov or equivalent sponsor system usage (e.g., NIH ASSIST/eRA Commons, NSF Research.gov, DoD eBRAP).
    10) indirect_cost_rate_info: Official indirect cost rate information and any mention of acknowledging or aligning to a 2025 15% cap where applicable (if present in the answer).
    11) active_federal_funding_record: Evidence of current or recent federal awards (can be NIH RePORTER, NSF awards page, or official university awards pages).
    12) research_security_training: Evidence of research security training requirements aligned with 2025 federal requirements for key personnel.

    13) primary_advisor_degree_requirement: Requirement that primary dissertation advisors hold doctoral degrees in relevant fields.
    14) credit_hours_requirement: Minimum credit hour and/or coursework requirements for degree completion (e.g., "minimum 60 credits", "core coursework list").
    15) comprehensive_exam_requirement: Evidence of comprehensive/qualifying/preliminary/candidacy exam prior to dissertation stage.

    RULES:
    - Extract only URLs explicitly present in the answer.
    - If a required field is missing from the answer, set it to null (or an empty array for sources).
    - Do not infer or generate new URLs.
    - Maintain the structure and field names exactly.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_urls(*url_lists: Optional[List[str]]) -> List[str]:
    """Deduplicate and filter URLs, keep only http(s) URLs."""
    seen = set()
    result: List[str] = []
    for urls in url_lists:
        if not urls:
            continue
        for u in urls:
            if not isinstance(u, str):
                continue
            u = u.strip()
            if not u:
                continue
            if not (u.startswith("http://") or u.startswith("https://")):
                continue
            if u not in seen:
                seen.add(u)
                result.append(u)
    return result


def _safe_str(x: Optional[str]) -> str:
    return (x or "").strip()


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_identity_checks(
    evaluator: Evaluator,
    parent_node,
    identity: Optional[ProgramIdentity],
) -> None:
    """
    Build verification for 'identifies_university_and_program':
    - Existence of names
    - Program is a doctoral PhD
    - Field is biomedical/biochemistry/molecular biology or closely related
    - University is US-based
    """
    node = evaluator.add_parallel(
        id="identifies_university_and_program",
        desc="Response states the university name (US-based) and the specific doctoral (PhD) program name in biomedical sciences/biochemistry/molecular biology or a closely related biological/health sciences field.",
        parent=parent_node,
        critical=True,
    )

    uni_name = _safe_str(identity.university_name if identity else None)
    prog_name = _safe_str(identity.program_name if identity else None)

    # Existence check: names provided
    evaluator.add_custom_node(
        result=(bool(uni_name) and bool(prog_name)),
        id="university_and_program_names_provided",
        desc="University and program names are provided in the response.",
        parent=node,
        critical=True,
    )

    # Prepare common sources
    common_sources = _normalize_urls(
        identity.identity_sources if identity else None,
        [identity.program_page_url] if identity and identity.program_page_url else None,
    )

    overview_sources = _normalize_urls(
        identity.university_overview_sources if identity else None,
        identity.identity_sources if identity else None,
        [identity.program_page_url] if identity and identity.program_page_url else None,
    )

    # Program is a PhD (doctoral) program
    prog_phd_leaf = evaluator.add_leaf(
        id="program_is_phd_doctoral",
        desc="The identified program is a doctoral (PhD) program.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The program '{prog_name}' at '{uni_name}' is a doctoral PhD program.",
        node=prog_phd_leaf,
        sources=common_sources,
        additional_instruction="Check the official program page or graduate catalog for explicit indications of 'PhD', 'Doctor of Philosophy', or doctoral program status.",
    )

    # Field classification requirement: biomedical-related
    field_leaf = evaluator.add_leaf(
        id="program_field_is_biomedical_related",
        desc="The program is in biomedical sciences, biochemistry, molecular biology, or a closely related biological/health sciences field.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The program fits into biomedical sciences, biochemistry, molecular biology, or a closely related biological/health sciences field.",
        node=field_leaf,
        sources=common_sources,
        additional_instruction="Use the official program description (name, overview, curriculum) to judge fit. Accept synonymous labels (e.g., Molecular & Cell Biology, Biochemistry & Molecular Biology, Biomedical Science umbrella) if clearly biological/health sciences.",
    )

    # US-based university requirement
    us_leaf = evaluator.add_leaf(
        id="university_is_us_based",
        desc="The university is located in the United States.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The university '{uni_name}' is located in the United States.",
        node=us_leaf,
        sources=overview_sources,
        additional_instruction="Look for address/location, 'United States', or state references on official pages. University domain and official 'About' or contact pages can serve as evidence.",
    )


async def build_sequential_requirement(
    evaluator: Evaluator,
    parent_node,
    id_prefix: str,
    description: str,
    evidence: Evidence,
    claim: str,
    add_ins: str,
) -> None:
    """
    Generic builder for a single requirement:
    - Sequential node (critical)
    - Custom existence check for sources
    - Claim verification by URLs
    """
    node = evaluator.add_sequential(
        id=id_prefix,
        desc=description,
        parent=parent_node,
        critical=True,
    )

    srcs = _normalize_urls(evidence.sources)

    # Existence of official sources
    evaluator.add_custom_node(
        result=(len(srcs) > 0),
        id=f"{id_prefix}_sources_present",
        desc=f"Official-source URLs are provided for: {description}",
        parent=node,
        critical=True,
    )

    # Verify claim with sources
    leaf = evaluator.add_leaf(
        id=f"{id_prefix}_supported",
        desc=description,
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction=add_ins,
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the single US PhD program task with comprehensive program-structure
    and research administration checks, grounded in official sources.
    """
    # Initialize evaluator with a non-critical root, and add a critical child node to represent the rubric root
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

    # Extract structured program info and sources from the answer
    program_data = await evaluator.extract(
        prompt=prompt_extract_program_info(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction",
    )

    # Add a critical aggregator representing the rubric root
    rubric_root = evaluator.add_parallel(
        id="evaluate_single_us_phd_program_response",
        desc="Evaluate whether the response identifies one US university PhD program in the specified fields and provides the required program-structure and research-administration information, supported by official sources.",
        parent=root,
        critical=True,
    )

    # Record task constraints as ground truth info for transparency
    evaluator.add_ground_truth({
        "constraints": [
            "US university; doctoral (PhD) program in biomedical sciences/biochemistry/molecular biology or closely related biological/health sciences.",
            "Typical time-to-degree completion within 4–7 years.",
            "Dissertation committees with at least 3 members.",
            "Committee members must hold doctoral/terminal degrees.",
            "At least some committee members must be full-time faculty.",
            "University has an IRB for human-subjects research and conducts continuing review.",
            "Eligible to receive federal research grants (e.g., NIH/NSF).",
            "Uses Grants.gov or equivalent sponsor systems (e.g., NIH ASSIST/eRA Commons, NSF Research.gov, DoD eBRAP).",
            "Indirect cost rate information includes acknowledgment of 2025 15% cap where applicable.",
            "Evidence of active federal funding record (current or recent awards).",
            "Research security training requirements consistent with 2025 federal requirements.",
            "Primary dissertation advisors hold doctoral degrees in relevant fields.",
            "Minimum credit hours and/or coursework requirements present.",
            "Comprehensive/qualifying/preliminary/candidacy exam before dissertation stage."
        ]
    })

    # 1) Identity checks
    await build_identity_checks(evaluator, rubric_root, program_data.identity)

    # 2) Time-to-degree requirement (4–7 years)
    await build_sequential_requirement(
        evaluator=evaluator,
        parent_node=rubric_root,
        id_prefix="time_to_degree_requirement",
        description="Provides an official-source citation showing the program’s typical time-to-degree completion is within 4–7 years.",
        evidence=program_data.time_to_degree,
        claim="The program’s typical time-to-degree completion is within 4–7 years.",
        add_ins="Look for 'typical', 'average', or 'usual' time-to-degree phrases. Accept ranges that fall inside 4–7 years or single values within 4–7 years.",
    )

    # 3) Dissertation committee minimum size
    await build_sequential_requirement(
        evaluator=evaluator,
        parent_node=rubric_root,
        id_prefix="dissertation_committee_min_size",
        description="Provides an official-source citation showing the program requires dissertation committees with at least 3 members.",
        evidence=program_data.committee_min_size,
        claim="The program requires dissertation committees with at least 3 members.",
        add_ins="Accept phrasing like 'minimum of three', '3 or more', 'at least three'. Verify policy or graduate handbook statements.",
    )

    # 4) Committee terminal degree requirement
    await build_sequential_requirement(
        evaluator=evaluator,
        parent_node=rubric_root,
        id_prefix="committee_terminal_degree_requirement",
        description="Provides an official-source citation showing dissertation committee members must hold doctoral degrees or appropriate terminal degrees.",
        evidence=program_data.committee_terminal_degree,
        claim="Dissertation committee members must hold doctoral degrees or appropriate terminal degrees.",
        add_ins="Accept PhD, MD, or other recognized terminal degrees appropriate for the field. Verify wording on official policy pages.",
    )

    # 5) Committee full-time faculty requirement
    await build_sequential_requirement(
        evaluator=evaluator,
        parent_node=rubric_root,
        id_prefix="committee_full_time_faculty_requirement",
        description="Provides an official-source citation showing at least some dissertation committee members must be full-time faculty at the institution.",
        evidence=program_data.committee_full_time_requirement,
        claim="At least some dissertation committee members must be full-time faculty at the institution.",
        add_ins="Accept requirements such as a majority internal full-time faculty, or minimum number of full-time faculty members on the committee.",
    )

    # 6) IRB exists for human subjects
    await build_sequential_requirement(
        evaluator=evaluator,
        parent_node=rubric_root,
        id_prefix="irb_exists_for_human_subjects",
        description="Provides an official-source citation showing the university has an IRB process for reviewing research involving human subjects.",
        evidence=program_data.irb_exists,
        claim="The university has an IRB process for reviewing research involving human subjects.",
        add_ins="Accept Human Research Protection Program (HRPP) pages, IRB overview pages, or official policy documents.",
    )

    # 7) IRB continuing review process
    await build_sequential_requirement(
        evaluator=evaluator,
        parent_node=rubric_root,
        id_prefix="irb_continuing_review_process",
        description="Provides an official-source citation showing the university’s IRB has a continuing review process for approved human-subjects studies.",
        evidence=program_data.irb_continuing_review,
        claim="The university’s IRB has a continuing review (renewal/ongoing oversight) process for approved human-subjects studies.",
        add_ins="Accept 'continuing review', 'annual renewal', 'ongoing approval', or similar oversight requirements.",
    )

    # 8) Federal grant eligibility
    await build_sequential_requirement(
        evaluator=evaluator,
        parent_node=rubric_root,
        id_prefix="federal_grant_eligibility",
        description="Provides an official-source citation showing the university is eligible to receive federal research grants (e.g., NIH/NSF).",
        evidence=program_data.federal_grant_eligibility,
        claim="The university is eligible to receive federal research grants from agencies such as NIH or NSF.",
        add_ins="Evidence may include sponsor eligibility statements, SAM/UEI registration info, or official references to NIH/NSF grant activity.",
    )

    # 9) Grant submission system
    await build_sequential_requirement(
        evaluator=evaluator,
        parent_node=rubric_root,
        id_prefix="grant_submission_system",
        description="Provides an official-source citation showing the university uses Grants.gov or an equivalent system for federal grant application submissions.",
        evidence=program_data.grant_submission_system,
        claim="The university uses Grants.gov or equivalent sponsor systems for federal grant submissions.",
        add_ins="Accept NIH ASSIST/eRA Commons, NSF Research.gov, DoD eBRAP, NASA NSPIRES, or similar systems as 'equivalent' to Grants.gov.",
    )

    # 10) Indirect cost rate information (including 2025 15% cap acknowledgment)
    await build_sequential_requirement(
        evaluator=evaluator,
        parent_node=rubric_root,
        id_prefix="indirect_cost_rate_information",
        description="Provides an official-source citation with the university’s indirect cost rate information for federal grants, including acknowledgment of the 2025 15% cap (as required by the constraints).",
        evidence=program_data.indirect_cost_rate_info,
        claim="The university provides indirect cost rate information for federal grants, including acknowledgment of a 2025 15% cap where applicable.",
        add_ins="Verify the presence of indirect/F&A rate information on official pages. For the 2025 15% cap, accept explicit acknowledgment or official policy adoption where applicable.",
    )

    # 11) Active federal funding record
    await build_sequential_requirement(
        evaluator=evaluator,
        parent_node=rubric_root,
        id_prefix="active_federal_funding_record",
        description="Provides official-source evidence of an active research funding record (e.g., current or recent federal grants).",
        evidence=program_data.active_federal_funding_record,
        claim="There is official-source evidence of current or recent federal awards for the university.",
        add_ins="Accept NIH RePORTER/NSF awards pages or official university award listings that show current or recent federal grants.",
    )

    # 12) Research security training requirement
    await build_sequential_requirement(
        evaluator=evaluator,
        parent_node=rubric_root,
        id_prefix="research_security_training_requirement",
        description="Provides an official-source citation showing the university has or is implementing research security training requirements for key personnel consistent with 2025 federal requirements (per constraints).",
        evidence=program_data.research_security_training,
        claim="The university has or is implementing research security training requirements for key personnel consistent with 2025 federal requirements.",
        add_ins="Accept official 'Research Security' pages, training policy documents, compliance office announcements indicating alignment with 2025 federal guidance.",
    )

    # 13) Primary advisor doctoral degree requirement
    await build_sequential_requirement(
        evaluator=evaluator,
        parent_node=rubric_root,
        id_prefix="primary_advisor_doctoral_degree_requirement",
        description="Provides an official-source citation showing faculty serving as primary dissertation advisors must hold doctoral degrees in relevant fields.",
        evidence=program_data.primary_advisor_degree_requirement,
        claim="Faculty serving as primary dissertation advisors must hold doctoral degrees in relevant fields.",
        add_ins="Verify advisor eligibility criteria in graduate handbooks or program policies. Accept PhD/MD in relevant disciplines.",
    )

    # 14) Minimum credit hours and/or coursework
    await build_sequential_requirement(
        evaluator=evaluator,
        parent_node=rubric_root,
        id_prefix="minimum_credit_hours_or_coursework",
        description="Provides an official-source citation of minimum credit hour and/or coursework requirements for degree completion.",
        evidence=program_data.credit_hours_requirement,
        claim="The program specifies minimum credit hour and/or coursework requirements for degree completion.",
        add_ins="Verify official curriculum/handbook pages for minimum credits or required coursework lists.",
    )

    # 15) Comprehensive/qualifying examination requirement
    await build_sequential_requirement(
        evaluator=evaluator,
        parent_node=rubric_root,
        id_prefix="comprehensive_or_qualifying_exam",
        description="Provides an official-source citation showing the program includes a comprehensive/qualifying examination requirement before the dissertation stage.",
        evidence=program_data.comprehensive_exam_requirement,
        claim="The program includes a comprehensive/qualifying/preliminary/candidacy examination before the dissertation stage.",
        add_ins="Accept synonymous terms (comprehensive, qualifying, preliminary, candidacy). Verify official program or graduate handbook pages.",
    )

    # Return structured evaluation summary
    return evaluator.get_summary()